"""Optimize Ry/CZ circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _circuit_state(params, n, layers):
    """Compute circuit state vector efficiently."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
            idx1 = idx0 | mask
            new_state = state.copy()
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        if layer < layers - 1:
            for q in range(n - 1):
                m1 = 1 << q
                m2 = 1 << (q + 1)
                both = np.array([i for i in range(dim) if (i & m1) and (i & m2)], dtype=np.int64)
                state[both] = -state[both]
    
    return state


def _energy(params, n, layers, coupling, field, H, e_ground):
    """Compute mean energy gap per spin."""
    state = _circuit_state(params, n, layers)
    e_circuit = np.real(np.vdot(state, H @ state))
    return (e_circuit - e_ground) / n


def _gradient(params, n, layers, coupling, field, H, e_ground):
    """Compute gradient of mean energy gap per spin using parameter-shift rule."""
    dim = 1 << n
    grad = np.zeros((layers, n), dtype=np.float64)
    
    for layer in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_plus[layer, q] += np.pi / 2
            s_plus = _circuit_state(p_plus, n, layers)
            e_plus = np.real(np.vdot(s_plus, H @ s_plus))
            
            p_minus = params.copy()
            p_minus[layer, q] -= np.pi / 2
            s_minus = _circuit_state(p_minus, n, layers)
            e_minus = np.real(np.vdot(s_minus, H @ s_minus))
            
            # Gradient of (e_circuit - e_ground)/n = (e_plus - e_minus)/(2*pi*n)
            grad[layer, q] = (e_plus - e_minus) / (2.0 * np.pi * n)
    
    return grad


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 115.0
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e_ground = eigvals[0]
    
    def obj(params):
        return _energy(params, n, layers, coupling, field, H, e_ground)
    
    def grad(params):
        return _gradient(params, n, layers, coupling, field, H, e_ground)
    
    ratio = field / coupling
    
    # Generate diverse initializations
    np.random.seed(42)
    initial_guesses = []
    
    # Mean-field based angle
    if ratio < 0.5:
        base = 0.3 * ratio
    elif ratio < 1.0:
        base = 0.6 * np.arctan(ratio)
    elif ratio < 2.0:
        base = np.pi / 4
    else:
        base = 0.4 * np.pi
    
    # Uniform
    initial_guesses.append(np.full((layers, n), base))
    
    # Alternating
    alt = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            alt[l, q] = base if q % 2 == 0 else -base
    initial_guesses.append(alt)
    
    # Zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # Pi/2
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    # Layer-dependent
    for l in range(layers):
        scale = 0.6 + 0.4 * (l / max(layers - 1, 1))
        initial_guesses.append(np.full((layers, n), base * scale))
    
    # Random perturbations
    for i in range(8):
        g = np.full((layers, n), base) + np.random.normal(0, 0.5, (layers, n))
        initial_guesses.append(g)
    
    # Random uniform
    for i in range(6):
        g = np.random.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        initial_guesses.append(g)
    
    # Small and large angles
    initial_guesses.append(np.full((layers, n), 0.1))
    initial_guesses.append(np.full((layers, n), 0.8))
    initial_guesses.append(np.full((layers, n), -0.5))
    
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 15:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def grad_flat(xf):
                p = xf.reshape((layers, n))
                return grad(p).flatten()
            
            # L-BFGS-B with correct gradient
            try:
                res = minimize(obj_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            # Nelder-Mead polish if time allows
            if time.time() > deadline - 8:
                continue
            
            try:
                res2 = minimize(obj_flat, best_params.flatten(), method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
            
            # Final L-BFGS-B polish
            if time.time() > deadline - 4:
                continue
            
            try:
                res3 = minimize(obj_flat, best_params.flatten(), jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        # Fallback: gradient descent
        for guess in initial_guesses:
            if time.time() > deadline - 10:
                break
            params = guess.copy()
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                g = grad(params)
                lr = 0.1 / (1 + step * 0.2)
                params_new = params - lr * g
                val_new = obj(params_new)
                if val_new < best_val:
                    best_val = val_new
                    best_params = params_new
                params = params_new
    
    return best_params
# EVOLVE-BLOCK-END
