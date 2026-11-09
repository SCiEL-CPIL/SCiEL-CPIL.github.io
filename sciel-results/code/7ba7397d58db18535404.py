import numpy as np
import time
from scipy.optimize import minimize

# Precomputed index caches for fast gate application
_IDX_CACHE = {}

def _get_idx_cache(n):
    """Get or build precomputed index arrays for fast gate application."""
    if n not in _IDX_CACHE:
        dim = 1 << n
        ry_cache = []
        for q in range(n):
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
            idx1 = idx0 | mask
            ry_cache.append((idx0, idx1))
        
        cz_cache = []
        for q in range(n - 1):
            mask1 = 1 << q
            mask2 = 1 << (q + 1)
            idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
            cz_cache.append(idx_both)
        
        _IDX_CACHE[n] = {'ry': ry_cache, 'cz': cz_cache}
    return _IDX_CACHE[n]

def _circuit_state(n, layers, params):
    """Compute state from circuit using precomputed indices."""
    cache = _get_idx_cache(n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
    return state

def _hamiltonian(n, coupling, field):
    """Build Hamiltonian."""
    dim = 1 << n
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

def _closed_form_angle(coupling, field):
    """
    Closed-form optimal Ry angle for Ising model.
    """
    if coupling == 0:
        return np.pi / 2
    ratio = field / coupling
    ratio = np.clip(ratio, -1e6, 1e6)
    return 2.0 * np.arctan(ratio)

def _energy_and_grad(n, layers, params_flat, H, e_ground):
    """Compute energy and gradient using parameter-shift rule."""
    dim = 1 << n
    params = params_flat.reshape((layers, n))
    
    # Compute energy
    state = _circuit_state(n, layers, params)
    energy = np.vdot(state, H @ state).real
    obj_val = (energy - e_ground) / n
    
    # Compute gradient using parameter-shift rule
    grad = np.zeros_like(params)
    shift = np.pi / 2
    
    for l in range(layers):
        for q in range(n):
            # Forward shift
            p_plus = params.copy()
            p_plus[l, q] += shift
            st_plus = _circuit_state(n, layers, p_plus)
            e_plus = np.vdot(st_plus, H @ st_plus).real
            
            # Backward shift
            p_minus = params.copy()
            p_minus[l, q] -= shift
            st_minus = _circuit_state(n, layers, p_minus)
            e_minus = np.vdot(st_minus, H @ st_minus).real
            
            grad[l, q] = (e_plus - e_minus) / (2 * shift * n)
    
    return obj_val, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 150s limit, use 145s
    
    # Compute closed-form optimal angle
    base_angle = _closed_form_angle(coupling, field)
    
    # Build Hamiltonian and compute ground state energy
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    def obj_grad(xf):
        val, grad = _energy_and_grad(n, layers, xf, H, e_ground)
        return val, grad
    
    # Generate diverse initial points
    inits = []
    
    # Closed-form uniform
    inits.append(np.full((layers, n), base_angle))
    
    # Uniform variations
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        inits.append(np.full((layers, n), base_angle * scale))
    
    # Layer-decaying patterns
    for decay in [0.7, 0.85, 1.0]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (decay ** l)
        inits.append(v)
    
    # Per-qubit variations around base
    rng = np.random.default_rng(42)
    for _ in range(3):
        v = np.full((layers, n), base_angle)
        v += rng.normal(0, 0.1, size=(layers, n))
        inits.append(v)
    
    # Evaluate all initial points and keep the best few
    scored_inits = []
    for v in inits:
        if time.time() > deadline - 90:
            break
        val = obj(v.flatten())
        scored_inits.append((val, v))
    
    scored_inits.sort(key=lambda x: x[0])
    
    best_val = scored_inits[0][0]
    best_params = scored_inits[0][1].copy()
    
    # Optimize from top 2 starting points using L-BFGS-B with gradient
    top_k = min(2, len(scored_inits))
    for k in range(top_k):
        if time.time() > deadline - 40:
            break
        
        start_x = scored_inits[k][1].flatten()
        
        # L-BFGS-B with gradient for faster convergence
        try:
            res = minimize(obj_grad, start_x, method='L-BFGS-B',
                         options={'maxiter': 300, 'gtol': 1e-12, 'maxfun': 10000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
        
        # Nelder-Mead for local refinement if L-BFGS-B didn't converge well
        if time.time() > deadline - 20:
            continue
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Ensure finite output
    if not np.all(np.isfinite(best_params)):
        best_params = np.full((layers, n), base_angle)
    
    return best_params
