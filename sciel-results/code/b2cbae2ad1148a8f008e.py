"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Caches for Hamiltonian and ground state data
_H_cache = {}
_e0_cache = {}
_circuit_indices_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms: -field * X_i
    for i in range(n):
        mask = 1 << i
        for state in range(dim):
            flipped = state ^ mask
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _get_circuit_indices(n, layers):
    """Pre-compute index pairs for fast state evolution."""
    key = (n, layers)
    if key in _circuit_indices_cache:
        return _circuit_indices_cache[key]
    
    dim = 1 << n
    idx = np.arange(dim)
    
    # Pre-compute indices for Ry(q)
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_indices.append((idx_0, idx_1))
        
    # Pre-compute indices for CZ(i, i+1)
    cz_indices = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_indices.append(both_set)
        
    _circuit_indices_cache[key] = (ry_indices, cz_indices)
    return _circuit_indices_cache[key]

def _circuit_state(n, thetas, layers):
    """Fast circuit state computation using pre-computed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = ry_indices[q]
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask_idx = cz_indices[i]
                state[mask_idx] *= -1.0
    
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)

    e0 = _exact_ground_energy(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)

    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Generate initial points
    inits = []
    
    # 1. Zero (Ferromagnetic limit)
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2 (Paramagnetic limit)
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Physics-informed mean-field estimate
    # tan(theta/2) ~ field/coupling
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full(layers * n, theta_mf))
    
    # 4. Alternating pattern (Antiferromagnetic hint)
    alt = np.zeros(layers * n).reshape(layers, n)
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = theta_mf
            else:
                alt[l, q] = -theta_mf
    inits.append(alt.flatten())
    
    # 5. Random restarts
    # Number of restarts depends on n
    num_random = 25 if n <= 4 else 15
    
    # Use a fixed seed for reproducibility in this context? No, evaluator might call multiple times.
    # But for a single call, we just use default randomness.
    for i in range(num_random):
        if time.time() > deadline - 20:
            break
        # Mix of uniform and gaussian around physics-informed center
        if i % 2 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        else:
            # Center around the mean-field estimate
            x_init = theta_mf + np.random.normal(0, 0.8, size=(layers * n))
        
        inits.append(x_init)
        
    # Sort inits by objective value to prioritize good starts
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 15:
            break
        try:
            v = obj(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from each start
    for v_init, x_init in init_objs:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Final polish with Nelder-Mead if time remains
    # Nelder-Mead can help escape shallow local minima
    if time.time() < deadline - 10.0:
        try:
            res_nm = minimize(obj, best_x, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass
            
    # One more L-BFGS-B polish
    if time.time() < deadline - 2.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
