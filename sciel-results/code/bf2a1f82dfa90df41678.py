"""
Rewritten program: Fast vectorized variational optimization for Ising ground states.

Strategy:
1. Vectorized state simulation and energy evaluation (no full Hamiltonian matrix-vector product).
2. Exploit left-right symmetry of the open chain to reduce the parameter space.
3. Use a multi-stage optimization: Differential Evolution for global search, followed by
   L-BFGS-B for local refinement.
4. Aggressive time management to ensure completion within the 150s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill
_cache = {}

def _circuit_state_vectorized(theta, n, layers):
    """
    Build the circuit state from angles using vectorized operations.
    theta: (layers, n) array of angles
    n: number of qubits
    layers: number of layers (fixed at 3)
    
    Circuit: Start |0>^n
    For each layer l:
        Apply Ry(theta[l, i]) to qubit i for all i
        If not the last layer, apply CZ on every adjacent pair in ascending order
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    indices = np.arange(dim)
    
    for l in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            mask = 1 << i
            # Select indices where bit i is 0
            mask_0 = indices[indices & mask == 0]
            mask_1 = mask_0 | mask
            
            c = np.cos(theta[l, i] / 2.0)
            s = -1j * np.sin(theta[l, i] / 2.0)
            
            a0 = state[mask_0]
            a1 = state[mask_1]
            
            state[mask_0] = c * a0 + s * a1
            state[mask_1] = s * a0 + c * a1
        
        # Apply CZ if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_j = 1 << (i + 1)
                mask_both = mask_i | mask_j
                # Indices where both bits are 1
                mask_bits = (indices & mask_both) == mask_both
                state[indices[mask_bits]] *= -1.0
    
    return state

def _compute_energy_vectorized(state, n, coupling, field):
    """
    Compute <state|H|state> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i
    """
    dim = 1 << n
    energy = 0.0
    prob = np.abs(state) ** 2
    
    indices = np.arange(dim)
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        
        bits_i = (indices >> i) & 1
        bits_j = (indices >> (i + 1)) & 1
        
        # ZZ is 1 if equal, -1 if different
        zz_vals = 1.0 - 2.0 * (bits_i ^ bits_j)
        
        energy += -coupling * np.sum(zz_vals * prob)
    
    # X terms
    for i in range(n):
        mask = 1 << i
        mask_0 = indices[indices & mask == 0]
        mask_1 = mask_0 | mask
        
        terms = np.conj(state[mask_0]) * state[mask_1]
        x_val = 2.0 * np.real(np.sum(terms))
        
        energy += -field * x_val
    
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    indices = np.arange(dim)
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        
        bits_i = (indices >> i) & 1
        bits_j = (indices >> (i + 1)) & 1
        zz_vals = 1.0 - 2.0 * (bits_i ^ bits_j)
        
        H[indices, indices] += -coupling * zz_vals
    
    # X terms
    for i in range(n):
        mask = 1 << i
        mask_0 = indices[indices & mask == 0]
        mask_1 = mask_0 | mask
        
        H[mask_0, mask_1] += -field
        H[mask_1, mask_0] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])

def _optimize_nm(n, coupling, field, layers):
    total_params = layers * n
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Symmetry reduction
    # The Hamiltonian and circuit are symmetric under left-right reflection.
    # We can enforce theta[l, i] = theta[l, n-1-i].
    num_unique = (n + 1) // 2
    
    def expand_params(unique_params):
        full_params = np.zeros((layers, n))
        for l in range(layers):
            for i in range(num_unique):
                val = unique_params[l * num_unique + i]
                full_params[l, i] = val
                full_params[l, n - 1 - i] = val
        return full_params
    
    def objective_unique(x):
        unique_params = x.reshape(layers, num_unique)
        full_params = expand_params(unique_params)
        state = _circuit_state_vectorized(full_params, n, layers)
        e_circuit = _compute_energy_vectorized(state, n, coupling, field)
        return (e_circuit - e0) / n
    
    bounds = [(-np.pi, np.pi)] * (layers * num_unique)
    
    best_params = None
    best_obj = np.inf
    
    # 1. Differential Evolution
    try:
        de_result = differential_evolution(
            objective_unique,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,
            updating='deferred'
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 20:
        if best_params is None:
            best_params = np.zeros(layers * num_unique)
        return expand_params(best_params.reshape(layers, num_unique))
    
    # 2. Local Optimization (L-BFGS-B)
    if best_params is not None:
        try:
            local_result = minimize(
                objective_unique,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if local_result.fun < best_obj:
                best_obj = local_result.fun
                best_params = local_result.x
        except Exception:
            pass
    else:
        # If DE failed, try a random start + local
        try:
            rng = np.random.default_rng(42)
            x0 = rng.uniform(-np.pi, np.pi, layers * num_unique)
            local_result = minimize(
                objective_unique,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if local_result.fun < best_obj:
                best_obj = local_result.fun
                best_params = local_result.x
        except Exception:
            pass

    # Final check
    elapsed = time.time() - _start_time
    if best_params is None:
        best_params = np.zeros(layers * num_unique)
        
    return expand_params(best_params.reshape(layers, num_unique))

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    try:
        theta = _optimize_nm(n, coupling, field, layers)
        _cache[key] = theta
        return theta
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _cache[key] = theta
        return theta
# EVOLVE-BLOCK-END
