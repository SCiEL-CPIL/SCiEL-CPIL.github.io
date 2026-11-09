"""Optimize Ry/CZ circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_masks(n):
    """Precompute index masks for Ry and CZ operations."""
    dim = 1 << n
    ry_masks = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx1 = idx0 | mask
        ry_masks.append((idx0, idx1))
    
    cz_masks = []
    for q in range(n - 1):
        m1 = 1 << q
        m2 = 1 << (q + 1)
        both = np.array([i for i in range(dim) if (i & m1) and (i & m2)], dtype=np.int64)
        cz_masks.append(both)
    
    return ry_masks, cz_masks


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    for i in range(n):
        for b in range(dim):
            bf = b ^ (1 << i)
            if b < bf:
                H[b, bf] += -field
                H[bf, b] += -field
    
    return H


def _circuit_state(params, n, layers, ry_masks, cz_masks):
    """Compute circuit state vector."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_masks[q]
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        if layer < layers - 1:
            for q in range(n - 1):
                both = cz_masks[q]
                state[both] = -state[both]
    
    return state


def _energy_and_grad(params, n, layers, coupling, field, H, e_ground, ry_masks, cz_masks):
    """Compute energy gap and gradient using parameter-shift rule."""
    dim = 1 << n
    state = _circuit_state(params, n, layers, ry_masks, cz_masks)
    e_circuit = np.real(np.vdot(state, H @ state))
    gap = (e_circuit - e_ground) / n
    
    grad = np.zeros((layers, n), dtype=np.float64)
    for layer in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_plus[layer, q] += np.pi / 2
            s_plus = _circuit_state(p_plus, n, layers, ry_masks, cz_masks)
            e_plus = np.real(np.vdot(s_plus, H @ s_plus))
            
            p_minus = params.copy()
            p_minus[layer, q] -= np.pi / 2
            s_minus = _circuit_state(p_minus, n, layers, ry_masks, cz_masks)
            e_minus = np.real(np.vdot(s_minus, H @ s_minus))
            
            grad[layer, q] = ((e_plus - e_minus) / (2 * np.pi) - e_ground / n)
    
    return gap, grad


def _mean_field_angle(ratio):
    """Mean-field estimate for optimal Ry angle based on field/coupling ratio."""
    if ratio < 0.5:
        return 0.2 * ratio
    elif ratio < 1.0:
        return 0.5 * np.arctan(ratio)
    elif ratio < 1.5:
        return np.pi / 4
    elif ratio < 2.5:
        return 0.5 * (np.pi / 2 + np.arctan(1.0 / max(ratio - 1, 0.1)))
    else:
        return 0.45 * np.pi


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e_ground = eigvals[0]
    
    ry_masks, cz_masks = _build_masks(n)
    
    def obj(params):
        state = _circuit_state(params, n, layers, ry_masks, cz_masks)
        e = np.real(np.vdot(state, H @ state))
        return (e - e_ground) / n
    
    def grad(params):
        g = np.zeros((layers, n), dtype=np.float64)
        for layer in range(layers):
            for q in range(n):
                p_p = params.copy()
                p_p[layer, q] += np.pi / 2
                sp = _circuit_state(p_p, n, layers, ry_masks, cz_masks)
                ep = np.real(np.vdot(sp, H @ sp))
                
                p_m = params.copy()
                p_m[layer, q] -= np.pi / 2
                sm = _circuit_state(p_m, n, layers, ry_masks, cz_masks)
                em = np.real(np.vdot(sm, H @ sm))
                
                g[layer, q] = (ep - em) / (2 * np.pi) - e_ground / n
        return g
    
    ratio = field / coupling
    base_angle = _mean_field_angle(ratio)
    
    np.random.seed(12345)
    
    initial_guesses = []
    
    # Physics-based: uniform angle
    initial_guesses.append(np.full((layers, n), base_angle))
    
    # Alternating pattern
    alt = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            alt[l, q] = base_angle if q % 2 == 0 else -base_angle
    initial_guesses.append(alt)
    
    # Zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # Pi/2
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    # Layer-dependent: smaller angles in early layers
    layer_dep = np.zeros((layers, n))
    for l in range(layers):
        scale = 0.7 + 0.3 * (l / max(layers - 1, 1))
        layer_dep[l] = base_angle * scale
    initial_guesses.append(layer_dep)
    
    # Random perturbations around base
    for i in range(6):
        g = np.full((layers, n), base_angle) + np.random.normal(0, 0.4, (layers, n))
        initial_guesses.append(g)
    
    # Random uniform
    for i in range(4):
        g = np.random.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        initial_guesses.append(g)
    
    # Small angles
    initial_guesses.append(np.full((layers, n), 0.1))
    
    # Large angles
    initial_guesses.append(np.full((layers, n), 0.8))
    
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 20:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def grad_flat(xf):
                p = xf.reshape((layers, n))
                return grad(p).flatten()
            
            # L-BFGS-B
            try:
                res = minimize(obj_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            # Nelder-Mead polish
            if time.time() > deadline - 10:
                break
            
            try:
                res2 = minimize(obj_flat, best_params.flatten(), method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
            
            # Final L-BFGS-B polish
            if time.time() > deadline - 5:
                break
            
            try:
                res3 = minimize(obj_flat, best_params.flatten(), jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        for guess in initial_guesses:
            if time.time() > deadline - 10:
                break
            params = guess.copy()
            for step in range(80):
                if time.time() > deadline - 5:
                    break
                g = grad(params)
                lr = 0.05 / (1 + step * 0.15)
                params_new = params - lr * g
                val_new = obj(params_new)
                if val_new < best_val:
                    best_val = val_new
                    best_params = params_new
                params = params_new
    
    if time.time() < deadline - 1:
        try:
            final_val = obj(best_params)
            if final_val < best_val:
                best_val = final_val
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
