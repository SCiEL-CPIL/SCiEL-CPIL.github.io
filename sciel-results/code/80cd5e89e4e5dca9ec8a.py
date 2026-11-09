import numpy as np
import time
from scipy.optimize import minimize

def _build_idx_cache(n):
    """Precompute index arrays for fast gate application."""
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
        
    return {'ry': ry_cache, 'cz': cz_cache}

def _circuit_state(n, layers, params, idx_cache):
    """Compute state from circuit using precomputed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = idx_cache['ry'][q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        # Apply CZ between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                idx_both = idx_cache['cz'][q]
                state[idx_both] = -state[idx_both]
    
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

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute
    idx_cache = _build_idx_cache(n)
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p, idx_cache)
        # Energy = <psi|H|psi>
        # H is real, st is complex
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    # Mean-field optimal angle
    ratio = field / coupling
    # sin(theta) = field*n / (2*coupling*(n-1)), clamped to [-1,1]
    s_target = ratio * n / (2 * (n - 1))
    if s_target > 1.0:
        base = np.pi / 2
    elif s_target < -1.0:
        base = -np.pi / 2
    else:
        base = np.arcsin(s_target)
    
    # Initial guesses
    guesses = []
    guesses.append(np.full((layers, n), base))
    guesses.append(np.full((layers, n), 0.0))
    guesses.append(np.full((layers, n), np.pi / 2))
    guesses.append(np.full((layers, n), -base))
    
    # Layer-dependent: first layer smaller, last layer closer to base
    for scale in [0.5, 0.8, 1.2]:
        g = np.zeros((layers, n))
        for l in range(layers):
            g[l, :] = base * (0.5 + 0.5 * l / max(layers - 1, 1)) * scale
        guesses.append(g)
    
    # Alternating
    g = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g[l, q] = base * (1 if q % 2 == 0 else -1)
    guesses.append(g)
    
    # Add a few random guesses
    np.random.seed(42)
    for _ in range(3):
        g = np.random.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
        
    best_params = guesses[0].copy()
    best_val = obj(best_params.flatten())
    
    # Optimize from each guess
    for g in guesses:
        if time.time() > deadline - 5:
            break
        x0 = g.flatten()
        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Polish from best
    if time.time() < deadline - 2:
        try:
            res = minimize(obj, best_params.flatten(), method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
