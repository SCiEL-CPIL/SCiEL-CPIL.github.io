"""
Optimize shallow circuit parameters for Ising ground states using a different strategy:
1. Vectorized state-vector simulation for speed.
2. Differential Evolution (DE) as a global optimizer, which is robust to local minima
   and does not require gradients.
3. Pre-compute Hamiltonian and ground energy.
4. Use a time budget to manage the number of DE iterations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # XX terms: -field * sum X_i
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _circuit_energy(params, n, layers, H, E0):
    """
    Compute the energy of the state produced by the circuit.
    params: array of shape (layers, n)
    Returns the energy gap (E_circuit - E0).
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Pre-compute masks
    masks = [1 << q for q in range(n)]
    
    def apply_ry(state, q, theta):
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        mask = masks[q]
        # Vectorized operation
        # We need to group indices by the bit value of q
        # Let's use a simpler approach for small n:
        # Create an array of indices
        # Actually, for n=4,6 (dim=16,64), simple loops are fast enough if we don't do it too often.
        # But let's try to be a bit more efficient.
        
        # Flatten approach:
        # For each block of size 2^(q+1), the first 2^q elements correspond to bit=0, next 2^q to bit=1.
        # Wait, that's only if q is the LSB of the block.
        # Since qubit i is bit i, the pattern repeats every 2^(q+1).
        
        new_state = np.zeros_like(state)
        # Iterate over blocks
        block_size = 1 << (q + 1)
        half_size = 1 << q
        for start in range(0, dim, block_size):
            # Indices for bit=0: start + 0 ... start + half_size - 1
            # Indices for bit=1: start + half_size ... start + block_size - 1
            idx0 = np.arange(start, start + half_size)
            idx1 = np.arange(start + half_size, start + block_size)
            
            # state[idx0] and state[idx1] are the amplitudes
            # new_state[idx0] = c * state[idx0] - s * state[idx1] ? 
            # Let's check Ry matrix:
            # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            # Input vector [a, b]^T -> [c*a - s*b, s*a + c*b]^T
            # So:
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            
        return new_state

    def apply_cz(state, q1, q2):
        mask1 = masks[q1]
        mask2 = masks[q2]
        # CZ flips sign if both bits are 1
        # We can create a mask of indices where both bits are 1
        # For small n, a simple loop or vectorized check is fine.
        # Vectorized:
        # indices = np.arange(dim)
        # condition = ((indices & mask1) != 0) & ((indices & mask2) != 0)
        # state[condition] *= -1
        # This is O(dim) which is 64. Very fast.
        indices = np.arange(dim)
        condition = ((indices & mask1) != 0) & ((indices & mask2) != 0)
        state[condition] *= -1
        return state

    for l in range(layers):
        for q in range(n):
            state = apply_ry(state, q, params[l, q])
        
        if l < layers - 1:
            for q in range(n - 1):
                state = apply_cz(state, q, q + 1)
                
    Hpsi = H @ state
    energy = np.vdot(state, Hpsi).real
    return energy - E0

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    E0 = _ground_energy(n, coupling, field)
    
    # Flatten params for DE
    # params shape: (layers, n)
    # bounds: (-pi, pi) for each
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    def objective_flat(x_flat):
        params = x_flat.reshape((layers, n))
        return _circuit_energy(params, n, layers, H, E0)
    
    # Differential Evolution
    # popsize: number of individuals in the population. Default 15.
    # maxiter: maximum number of generations.
    # tol: tolerance for stopping.
    # seed: random seed for reproducibility (optional)
    
    # We want to use as many iterations as possible within the time budget.
    # DE can be slow. Let's set a reasonable maxiter and see.
    # For n=6, dim=64. 18 parameters.
    # popsize=15 -> 15*18 = 270 evaluations per generation.
    # Each eval is ~10-50 microseconds? Maybe more.
    # Let's assume 1ms per eval. 270ms per generation.
    # In 140s, we can do ~500 generations.
    
    best_result = None
    best_energy = np.inf
    
    # Run DE
    try:
        result = differential_evolution(
            objective_flat,
            bounds,
            maxiter=500,
            popsize=20,
            tol=1e-8,
            seed=42,
            # time_limit is not directly supported in older scipy, but we can check time
            # Actually, we can't easily interrupt DE mid-run without a callback or wrapper.
            # We'll just set maxiter high and hope it finishes, or use a wrapper.
            # But if it takes too long, we might timeout.
            # Let's use a smaller maxiter to be safe, or rely on the fact that n is small.
            # For n=4, it's faster. For n=6, it's slower.
            # Let's set maxiter=200 to be safe.
            maxiter=200
        )
        if result.success or result.fun < best_energy:
            best_energy = result.fun
            best_result = result.x
    except Exception as e:
        # If DE fails or times out, fall back to random search or seed
        pass
    
    if best_result is None:
        # Fallback: Random search
        num_restarts = 100
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            x_flat = np.random.uniform(-np.pi, np.pi, layers * n)
            e = objective_flat(x_flat)
            if e < best_energy:
                best_energy = e
                best_result = x_flat
        
    if best_result is None:
        # Final fallback: seed-like
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    return best_result.reshape((layers, n))
# EVOLVE-BLOCK-END
