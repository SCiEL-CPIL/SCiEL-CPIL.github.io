"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Caches
_H_cache = {}
_e0_cache = {}
_gs_cache = {}
_circuit_indices_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
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

def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _gs_cache:
        return _gs_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    psi = eigvecs[:, idx].copy()
    # Fix global phase
    max_idx = np.argmax(np.abs(psi))
    if np.abs(psi[max_idx]) > 1e-12:
        phase = np.angle(psi[max_idx])
        psi = psi * np.exp(-1j * phase)
    _gs_cache[key] = psi
    return _gs_cache[key]

def _get_circuit_indices(n, layers):
    key = (n, layers)
    if key in _circuit_indices_cache:
        return _circuit_indices_cache[key]
    
    dim = 1 << n
    idx = np.arange(dim)
    
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_indices.append((idx_0, idx_1))
        
    cz_indices = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_indices.append(both_set)
        
    _circuit_indices_cache[key] = (ry_indices, cz_indices)
    return _circuit_indices_cache[key]

def _circuit_state(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = ry_indices[q]
            
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
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

def _gs_informed_init(n, coupling, field, layers):
    """Compute initial angles by matching ground state single-qubit marginals."""
    psi_gs = _exact_ground_state(n, coupling, field)
    dim = 1 << n
    
    # For each qubit, compute the probability of |1> in the ground state
    # This gives us a target for the first-layer Ry angles
    # After Ry(theta) on |0>, P(1) = sin^2(theta/2)
    # So theta = 2*arcsin(sqrt(p1))
    
    theta_init = np.zeros((layers, n))
    
    for q in range(n):
        mask = 1 << q
        # Count states where qubit q is 1
        p1 = 0.0
        for state in range(dim):
            if (state & mask) != 0:
                p1 += np.abs(psi_gs[state])**2
        p1 = np.clip(p1, 0.0, 1.0)
        theta_q = 2.0 * np.arcsin(np.sqrt(p1))
        
        # Use this for all layers as a starting point
        # The first layer is most important for setting the marginals
        for l in range(layers):
            theta_init[l, q] = theta_q
    
    return theta_init

def _alternating_gs_init(n, coupling, field, layers):
    """Alternating version of GS-informed init for antiferromagnetic hints."""
    base = _gs_informed_init(n, coupling, field, layers)
    alt = np.zeros_like(base)
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = base[l, q]
            else:
                alt[l, q] = -base[l, q]
    return alt

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    e0 = _exact_ground_energy(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)

    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Generate initial points
    inits = []
    
    # 1. Zero
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. GS-informed init (strongest)
    try:
        gs_init = _gs_informed_init(n, coupling, field, layers)
        inits.append(gs_init.flatten())
    except Exception:
        pass
    
    # 4. Alternating GS-informed
    try:
        alt_init = _alternating_gs_init(n, coupling, field, layers)
        inits.append(alt_init.flatten())
    except Exception:
        pass
    
    # 5. Mean-field estimate
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full(layers * n, theta_mf))
    
    # 6. Random restarts - fewer since we have good physics-based inits
    num_random = 12 if n <= 4 else 8
    for i in range(num_random):
        if time.time() > deadline - 25:
            break
        if i % 3 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        elif i % 3 == 1:
            # Perturb GS-informed init
            try:
                base = _gs_informed_init(n, coupling, field, layers).flatten()
                x_init = base + np.random.normal(0, 0.5, size=(layers * n))
            except Exception:
                x_init = theta_mf + np.random.normal(0, 0.8, size=(layers * n))
        else:
            x_init = theta_mf + np.random.normal(0, 0.8, size=(layers * n))
        inits.append(x_init)
    
    # Evaluate and sort inits
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 20:
            break
        try:
            v = obj(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from each start
    for v_init, x_init in init_objs:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Nelder-Mead polish
    if time.time() < deadline - 15:
        try:
            res_nm = minimize(obj, best_x, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.copy()
        except Exception:
            pass
            
    # Final L-BFGS-B polish
    if time.time() < deadline - 5:
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
