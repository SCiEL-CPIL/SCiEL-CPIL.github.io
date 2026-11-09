import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    I = np.eye(2, dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    # X terms
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

def _circuit_state(n, layers, params):
    """Compute the state vector from the circuit parameters."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            # For each pair of indices differing only in bit q
            for i in range(dim):
                if (i & mask) == 0:
                    j = i | mask
                    old0 = state[i]
                    old1 = state[j]
                    state[i] = c * old0 - s * old1
                    state[j] = s * old0 + c * old1
        
        # Apply CZ between layers (not after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                for i in range(dim):
                    if (i & mask1) and (i & mask2):
                        state[i] = -state[i]
    
    return state

def _energy_and_gradient(n, layers, params, H):
    """Compute energy and gradient using finite differences for robustness."""
    dim = 1 << n
    e0 = np.vdot(_circuit_state(n, layers, params), H @ _circuit_state(n, layers, params)).real
    grad = np.zeros_like(params)
    eps = 1e-6
    for l in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_plus[l, q] += eps
            e_plus = np.vdot(_circuit_state(n, layers, p_plus), H @ _circuit_state(n, layers, p_plus)).real
            p_minus = params.copy()
            p_minus[l, q] -= eps
            e_minus = np.vdot(_circuit_state(n, layers, p_minus), H @ _circuit_state(n, layers, p_minus)).real
            grad[l, q] = (e_plus - e_minus) / (2 * eps)
    return e0, grad

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
    
    return guesses, base

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    H = _build_hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    # Generate initial guesses
    guesses, base = _mean_field_angles(n, coupling, field, layers)
    
    # Add random guesses
    rng = np.random.RandomState(42)
    for _ in range(10):
        g = rng.uniform(-np.pi, np.pi, (layers, n))
        guesses.append(g)
    
    # Small perturbations around mean-field
    for _ in range(5):
        g = np.full((layers, n), base) + rng.normal(0, 0.15, (layers, n))
        guesses.append(g)
    
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
