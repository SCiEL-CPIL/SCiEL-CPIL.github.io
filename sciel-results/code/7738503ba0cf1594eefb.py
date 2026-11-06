"""
Strategy: Direct Algebraic Construction with Local Optimization
Instead of full gradient descent on the entire 18-parameter space (for n=6, 3 layers),
we exploit the structure of the 3-layer Ry/CZ circuit.

Circuit:
|0>^n 
-> Ry(theta_1) 
-> CZ_adjacent 
-> Ry(theta_2) 
-> CZ_adjacent 
-> Ry(theta_3)

We can view this as:
1. Prepare a state using the first two layers (Ry1, CZ, Ry2, CZ).
2. The final layer Ry(theta_3) is a tensor product of single-qubit rotations.
3. The energy E = <psi|H|psi> is a sum of terms.
   - X terms: <X_i> depends only on theta_3[i] and the state of qubit i before the last layer.
   - ZZ terms: <Z_i Z_{i+1}> depends on theta_3[i], theta_3[i+1] and the 2-qubit reduced density matrix of (i, i+1) before the last layer.

Since the final layer is local, we can optimize theta_3 for fixed theta_1, theta_2.
However, a full optimization of theta_1, theta_2 is still needed.
Given the small size (n=4,6), we can use a multi-start random search for theta_1, theta_2 
and analytically/optimize theta_3 efficiently.

Actually, even simpler: 
For n=4,6, the total parameters are 3*n = 12 or 18.
A standard L-BFGS-B from a good seed is fast. The previous failures likely stemmed from 
poor initialization or getting stuck in local minima.
This strategy uses a **grid-based initialization** combined with **L-BFGS-B**.
We will generate a few diverse initial points based on physical intuition 
(adiabatic-like paths or mean-field estimates) and run a short optimization from each, 
keeping the best result.

To ensure speed, we will use a pre-computed exact ground state energy for the given 
n, coupling, field to normalize the objective correctly, but the main cost is the 
circuit simulation. For n=6, the state vector is 2^6=64 complex numbers. 
Simulating the circuit is extremely fast.

We will implement a robust multi-start optimizer.
"""

import numpy as np
import time
from scipy.optimize import minimize

# Cache for exact ground energies to avoid recomputation
_exact_energy_cache = {}

def _get_exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _exact_energy_cache:
        return _exact_energy_cache[key]
    
    # Build Hamiltonian for open chain
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # Basis: |00...0> to |11...1>
    # Qubit i is bit i. 
    # Note: The problem statement says "qubit i is bit i". 
    # Standard convention: |q_{n-1} ... q_1 q_0>. 
    # If qubit i is bit i, then qubit 0 is LSB, qubit n-1 is MSB.
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # X terms
    for i in range(n):
        # Flip bit i
        mask = 1 << i
        for state in range(dim):
            target = state ^ mask
            H[target, state] -= field
            
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        for state in range(dim):
            # Z_i = +1 if bit i is 0, -1 if bit i is 1
            zi = 1 if (state & mask_i) == 0 else -1
            zj = 1 if (state & mask_j) == 0 else -1
            H[state, state] -= coupling * zi * zj
            
    # Diagonalize
    eigvals = np.linalg.eigvalsh(H)
    e0 = np.min(eigvals).real
    
    _exact_energy_cache[key] = e0
    return e0

def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit in state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Ry = [[c, -s], [s, c]]
    # We need to iterate over all states and update the amplitudes.
    # For efficiency, we reshape and use numpy operations.
    dim = state.shape[0]
    n_qubits = int(np.log2(dim))
    
    # Reshape to tensor
    # Axis order: qubit n-1 is axis 0, ..., qubit 0 is axis n-1?
    # Let's stick to a simple loop for clarity given small n, or vectorize.
    # Vectorized approach:
    # Create masks for the qubit.
    
    # Actually, for n=6, dim=64. A loop over 64 states is trivial.
    # But let's try to be slightly faster.
    
    new_state = np.zeros_like(state)
    for i in range(dim):
        # Check bit qubit
        if (i >> qubit) & 1 == 0:
            # |0> -> c|0> + s|1>
            target = i | (1 << qubit)
            new_state[i] += c * state[i]
            new_state[target] += s * state[i]
        else:
            # |1> -> -s|0> + c|1>
            target = i & ~(1 << qubit)
            new_state[i] += c * state[i]
            new_state[target] -= s * state[i]
            
    return new_state

def _apply_cz(state, q1, q2):
    """Apply CZ to qubits q1, q2."""
    dim = state.shape[0]
    new_state = np.zeros_like(state)
    mask1 = 1 << q1
    mask2 = 1 << q2
    
    for i in range(dim):
        if ((i & mask1) != 0) and ((i & mask2) != 0):
            new_state[i] = -state[i]
        else:
            new_state[i] = state[i]
            
    return new_state

def _circuit_energy(params, n, coupling, field):
    """
    params: array of shape (layers, n)
    Returns energy of the state produced by the circuit.
    """
    layers = params.shape[0]
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            state = _apply_ry(state, params[l, i], i)
        
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
                
    # Calculate Energy
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # E = <psi|H|psi>
    # We can compute this by applying H to psi and taking inner product.
    # Or compute expectations directly.
    # Direct expectation is faster for this small system.
    
    energy = 0.0
    prob = np.abs(state)**2
    
    # X terms: <X_i> = sum_{s, t} psi*_s X_i psi_t
    # X_i flips bit i.
    # <X_i> = 2 * Re( sum_{s where bit_i=0} psi*_s psi_{s ^ (1<<i)} )
    for i in range(n):
        mask = 1 << i
        sum_val = 0.0
        for s in range(2**n):
            if (s & mask) == 0:
                t = s ^ mask
                sum_val += np.conj(state[s]) * state[t]
        xi = 2.0 * np.real(sum_val)
        energy -= field * xi
        
    # ZZ terms: <Z_i Z_j> = sum_s psi*_s (Z_i Z_j) psi_s
    # Z_i Z_j is diagonal.
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        for s in range(2**n):
            zi = 1 if (s & mask_i) == 0 else -1
            zj = 1 if (s & mask_j) == 0 else -1
            energy -= coupling * zi * zj * prob[s]
            
    return np.real(energy)

def _objective(params_flat, n, coupling, field):
    params = params_flat.reshape(3, n)
    e_circuit = _circuit_energy(params, n, coupling, field)
    e_exact = _get_exact_ground_energy(n, coupling, field)
    gap = (e_circuit - e_exact) / n
    return gap

def _optimize_for_config(n, coupling, field, deadline):
    best_gap = np.inf
    best_params = np.zeros((3, n))
    
    # Multi-start strategy
    # Start 1: All zeros
    # Start 2: Pi/2 in last layer (product |+> state, which is good for strong field)
    # Start 3: Random
    
    starts = []
    
    # Start 0: Zeros
    starts.append(np.zeros((3, n)))
    
    # Start 1: Last layer Pi/2
    p1 = np.zeros((3, n))
    p1[-1] = np.pi / 2
    starts.append(p1)
    
    # Start 2: Random uniform
    np.random.seed(42) # Fixed seed for reproducibility within the run? 
    # Actually, let's use a few fixed random seeds to ensure diversity but determinism.
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        p_rand = np.random.uniform(-np.pi, np.pi, (3, n))
        starts.append(p_rand)
        
    # Start 3: Mean-field like
    # For strong field, angles close to pi/2.
    # For strong coupling, angles close to 0 or pi.
    # Let's try a linear interpolation based on field/coupling ratio.
    # If field >> coupling, we want |+> (pi/2).
    # If coupling >> field, we want |0> or |1> (0 or pi).
    # Simple heuristic:
    p_mf = np.zeros((3, n))
    for l in range(3):
        # Scale angle towards pi/2
        p_mf[l] = np.pi / 4 * (l + 1) / 3.0 
    starts.append(p_mf)
    
    for start_params in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective, 
                start_params.flatten(), 
                args=(n, coupling, field), 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            gap = res.fun
            if gap < best_gap:
                best_gap = gap
                best_params = res.x.reshape(3, n)
                
        except Exception:
            continue
            
    return best_params, best_gap

def angles(n, coupling, field, layers):
    start_time = time.time()
    # Safety margin: 145s
    deadline = start_time + 145.0
    
    # We assume layers=3 as per task description
    if layers != 3:
        # Fallback to simple product state if layers is not 3
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    params, gap = _optimize_for_config(n, coupling, field, deadline)
    
    # If we ran out of time, params might be the last one found or zeros.
    # The function returns the best found so far in the loop.
    
    return params
