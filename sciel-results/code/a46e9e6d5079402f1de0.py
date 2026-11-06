"""Improved optimizer for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
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

def _circuit_state_fast(n, thetas, layers):
    """Fast circuit state computation using in-place operations."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    idx = np.arange(dim)
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            mask = 1 << q
            mask_bit = (idx & mask) == 0
            idx_0 = idx[mask_bit]
            idx_1 = idx_0 + mask
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask1 = 1 << i
                mask2 = 1 << (i + 1)
                both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
                state[both_set] *= -1.0
    
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _physics_informed_init(n, coupling, field, layers):
    """Generate physics-informed initial guesses based on perturbation theory."""
    inits = []
    
    # Paramagnetic limit: field >> coupling, all qubits in |->
    inits.append(np.full((layers, n), np.pi / 2))
    
    # Ferromagnetic limit: coupling >> field, all qubits in |0>
    inits.append(np.zeros((layers, n)))
    
    # Intermediate: based on field strength
    # For transverse-field Ising model, the ground state transitions
    # from ferromagnetic (|0...0>) to paramagnetic (|->^n)
    
    # Estimate optimal angle from mean-field: tan(theta/2) ~ field/coupling
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full((layers, n), theta_mf))
    
    # Antiferromagnetic-like: alternating angles
    alt = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = theta_mf
            else:
                alt[l, q] = -theta_mf
    inits.append(alt)
    
    # Layer-dependent: compensate for CZ correlations
    # CZ creates correlations, so later layers may need different angles
    layer_comp = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            # Later layers slightly adjust to compensate for CZ
            layer_comp[l, q] = theta_mf * (1.0 + 0.1 * (l - 1))
    inits.append(layer_comp)
    
    # Small perturbations around key points
    for base in [0.0, np.pi / 2, theta_mf]:
        for delta in [-0.3, -0.1, 0.1, 0.3]:
            inits.append(np.full((layers, n), base + delta))
    
    return inits

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)
    
    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Generate diverse initial points
    inits = _physics_informed_init(n, coupling, field, layers)
    
    # Add random inits
    num_random = 20 if n <= 4 else 12
    for i in range(num_random):
        if time.time() > deadline - 15:
            break
        if i % 3 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        elif i % 3 == 1:
            x_init = np.pi / 2 + np.random.normal(0, 0.8, size=(layers * n))
        else:
            x_init = np.random.normal(0, 1.0, size=(layers * n))
        inits.append(x_init)
    
    # Sort inits by objective value to prioritize good starts
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 10:
            break
        try:
            v = obj(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from each start, prioritizing lower initial objective
    for v_init, x_init in init_objs:
        if time.time() > deadline - 5:
            break
        
        try:
            # Use tighter tolerances for better convergence
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15,
                                   'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # One more polish if time remains
    if time.time() < deadline - 1.0 and best_val > 0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
