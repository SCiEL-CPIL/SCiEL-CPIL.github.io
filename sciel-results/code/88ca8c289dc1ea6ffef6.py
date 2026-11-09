import numpy as np
import time
from scipy.optimize import minimize

# Precomputed masks for speed
_MASK_CACHE = {}

def _get_masks(n):
    """Precompute and cache index masks for Ry and CZ operations."""
    key = n
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    dim = 1 << n
    ry_masks = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.where((np.arange(dim) & mask) == 0)[0]
        idx1 = idx0 | mask
        ry_masks.append((idx0, idx1))
    cz_masks = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        both = np.where((np.arange(dim) & mask1) != 0)[0]
        both = both[(both & mask2) != 0]
        cz_masks.append(both)
    _MASK_CACHE[key] = (ry_masks, cz_masks)
    return _MASK_CACHE[key]

def _circuit_state(n, layers, params):
    """Compute state from circuit. params: (layers, n) array of angles."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    ry_masks, cz_masks = _get_masks(n)
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_masks[q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                both = cz_masks[q]
                state[both] = -state[both]
    
    return state

def _hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ terms
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            H[b, b] += -coupling * (1 - 2 * bi) * (1 - 2 * bj)
    # X terms
    for i in range(n):
        for b in range(dim):
            bf = b ^ (1 << i)
            if b < bf:
                H[b, bf] -= field
                H[bf, b] -= field
    return H

def _mean_field_angles(n, coupling, field, layers):
    """Generate mean-field inspired initial guesses."""
    ratio = field / coupling
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        base = np.pi / 2
    elif s_target < -1.0:
        base = -np.pi / 2
    else:
        base = np.arcsin(s_target)
    
    guesses = []
    # Uniform
    guesses.append(np.full((layers, n), base))
    # Zero
    guesses.append(np.zeros((layers, n)))
    # Pi/2
    guesses.append(np.full((layers, n), np.pi / 2))
    # -base
    guesses.append(np.full((layers, n), -base))
    
    # Layer-dependent
    for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.5 + 0.5 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating
    for sign_scale in [0.5, 0.8, 1.0]:
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = base * sign_scale * (1 if q % 2 == 0 else -1)
        guesses.append(g)
    
    # Small perturbations around mean-field
    rng = np.random.RandomState(42)
    for _ in range(5):
        g = np.full((layers, n), base) + rng.normal(0, 0.15, (layers, n))
        guesses.append(g)
    
    # Add random guesses
    for _ in range(10):
        g = rng.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
    
    return guesses

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        return (np.vdot(st, H @ st).real - e_ground) / n
    
    # Generate initial guesses
    guesses = _mean_field_angles(n, coupling, field, layers)
    
    # Evaluate and sort
    scored = []
    for g in guesses:
        v = obj(g.flatten())
        scored.append((v, g))
    scored.sort(key=lambda x: x[0])
    
    best_val = scored[0][0]
    best_params = scored[0][1].copy()
    
    # Optimize from top candidates
    max_candidates = min(len(scored), 8)
    for i in range(max_candidates):
        if time.time() > deadline - 20:
            break
        v0, g0 = scored[i]
        if v0 > best_val + 0.5:
            break
        
        x0 = g0.flatten()
        try:
            # L-BFGS-B with bounds
            bounds = [(-np.pi, np.pi)] * (layers * n)
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
        
        # Polish with Nelder-Mead
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Final polish if time remains
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
    
    return best_params
