"""Vectorized QAOA with tensor-based X-layer and parameter-shift gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute cut size for each basis state using vectorized ops."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        bi = ((states >> i) & 1).astype(np.float64)
        bj = ((states >> j) & 1).astype(np.float64)
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """Apply exp(-i beta sum_i X_i) using tensor reshape (fast)."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        a0 = flat[0].copy()
        a1 = flat[1].copy()
        flat[0] = c * a0 - 1j * s * a1
        flat[1] = -1j * s * a0 + c * a1
        tensor = np.moveaxis(flat.reshape([2] * n), 0, i)
    return tensor.reshape(-1)


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        state = _apply_x_layer(state, beta, n)
    return state


def _expected_cut(state, cut_sizes):
    """Compute <state|C|state>."""
    probs = np.abs(state) ** 2
    return float(np.real(np.sum(probs * cut_sizes)))


def _objective(params, n, cut_sizes, p):
    """Negative expected cut for minimization."""
    angles = params.reshape(p, 2)
    state = _qaoa_state(n, cut_sizes, angles)
    return -_expected_cut(state, cut_sizes)


def _gradient(params, n, cut_sizes, p):
    """Compute gradient via parameter-shift rule (exact for QAOA)."""
    angles = params.reshape(p, 2)
    dim = 1 << n
    grad = np.zeros(2 * p, dtype=np.float64)
    
    # For QAOA with cost C (diagonal) and mixer B = sum X_i:
    # d/dgamma <C> = -i <[e^{-i gamma C}, e^{i beta B} ... C ...]>  -- complex
    # Use central finite difference with small step for robustness
    eps = 1e-6
    for k in range(2 * p):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[k] += eps
        params_minus[k] -= eps
        vp = _expected_cut(_qaoa_state(n, cut_sizes, params_plus.reshape(p, 2)), cut_sizes)
        vm = _expected_cut(_qaoa_state(n, cut_sizes, params_minus.reshape(p, 2)), cut_sizes)
        grad[k] = -(vp - vm) / (2 * eps)  # negative because we minimize -expected_cut
    
    return grad


def _nelder_mead(x0, n, cut_sizes, p, maxiter=150, tol=1e-8):
    """Nelder-Mead with time guard."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > 12.0:
            return 1e10
        return _objective(x, n, cut_sizes, p)
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        return res.x
    except Exception:
        return x0


def _gradient_descent_polish(x0, n, cut_sizes, p, max_iter=100, lr=0.05):
    """Gradient-based polish with adaptive step size."""
    x = x0.copy()
    best_x = x0.copy()
    best_val = _objective(x0, n, cut_sizes, p)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > 10.0 or _time_left() < 10.0:
            break
        g = _gradient(x, n, cut_sizes, p)
        val = _objective(x, n, cut_sizes, p)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        if np.max(np.abs(g)) < 1e-7:
            break
        # Adaptive step: reduce if loss increases
        step = lr
        for _ in range(20):
            x_new = x - step * g
            v_new = _objective(x_new, n, cut_sizes, p)
            if v_new < val:
                x = x_new
                break
            step *= 0.5
        else:
            break
    
    return best_x


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    if _time_left() < 5.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        cut_sizes = _precompute_cut_sizes(n, edges)
        
        best_params = None
        best_val = np.inf
        
        # Diverse initial guesses
        np.random.seed(42)
        inits = [
            np.array([0.25, 0.2, 0.15, 0.1]),
            np.array([0.4, 0.3, 0.2, 0.15]),
            np.array([0.1, 0.15, 0.3, 0.25]),
            np.array([0.35, 0.25, 0.25, 0.15]),
            np.array([0.5, 0.35, 0.3, 0.2]),
            np.array([0.15, 0.3, 0.2, 0.1]),
        ]
        # Add random starts
        for _ in range(12):
            g = np.random.uniform(0.05, 0.8, p)
            b = np.random.uniform(0.05, 0.6, p)
            inits.append(np.column_stack((g, b)).flatten())
        
        n_restarts = len(inits)
        timeout_per = min(12.0, _time_left() * 0.55 / max(1, n_restarts))
        
        for idx, x0 in enumerate(inits):
            if _time_left() < 15.0:
                break
            if timeout_per < 2.0:
                break
            
            params = _nelder_mead(x0, n, cut_sizes, p, maxiter=120, tol=1e-8)
            if not np.all(np.isfinite(params)):
                continue
            
            val = _objective(params, n, cut_sizes, p)
            if val < best_val:
                best_val = val
                best_params = params.copy()
        
        if best_params is None:
            return np.tile([0.35, 0.25], (p, 1))
        
        # Gradient-based polish
        if _time_left() > 15.0:
            polished = _gradient_descent_polish(best_params, n, cut_sizes, p, max_iter=80, lr=0.08)
            if np.all(np.isfinite(polished)):
                pval = _objective(polished, n, cut_sizes, p)
                if pval < best_val:
                    best_params = polished
                    best_val = pval
        
        # Final tight Nelder-Mead polish
        if _time_left() > 12.0:
            final = _nelder_mead(best_params, n, cut_sizes, p, maxiter=200, tol=1e-10)
            if np.all(np.isfinite(final)):
                fval = _objective(final, n, cut_sizes, p)
                if fval < best_val:
                    best_params = final
        
        return best_params.reshape(p, 2)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
