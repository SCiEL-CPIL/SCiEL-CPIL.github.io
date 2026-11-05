"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _chebyshev_basis(N, order):
    """Return Chebyshev basis vectors on [-1/4, 1/4] mapped to [0, N-1]."""
    x = np.linspace(-1, 1, N)  # map interval to [-1,1] for Chebyshev
    basis = np.zeros((order, N))
    for k in range(order):
        basis[k] = np.cos(k * np.arccos(x))
    return basis

def _eval_value(h):
    """Compute the exact evaluator value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / s**2)

def _objective(coeffs, basis, N):
    """Objective: minimize the evaluator value for h = basis @ coeffs, normalized to sum=1."""
    h = basis @ coeffs
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    h = h / s  # normalize so sum = 1
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv))

def construct():
    global _DEADLINE
    if time.time() > _DEADLINE:
        return np.ones(N)
    
    # Use Chebyshev basis of moderate order
    order = 10
    basis = _chebyshev_basis(N, order)
    
    # Bounds for coefficients
    bounds = [(-5.0, 5.0)] * order
    
    # Initial best: seed
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # should be 2.0
    
    # Try a few quick random starts first
    np.random.seed(42)
    for _ in range(20):
        if time.time() > _DEADLINE:
            break
        coeffs = np.random.uniform(-3, 3, order)
        h = basis @ coeffs
        s = np.sum(h)
        if abs(s) < 1e-15:
            continue
        h = h / s
        val = _eval_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Differential evolution over Chebyshev coefficients
    if time.time() < _DEADLINE - 30:
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(basis, N),
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True
            )
            if result.success or result.fun < best_val:
                coeffs_opt = result.x
                h_opt = basis @ coeffs_opt
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-15:
                    h_opt = h_opt / s_opt
                    val_opt = _eval_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: local optimization on the best found
    if time.time() < _DEADLINE - 10 and best_val < 2.0:
        try:
            from scipy.optimize import minimize
            # Extract current coefficients in Chebyshev basis
            # Use least squares to get coefficients
            coeffs_init = np.linalg.lstsq(basis, best_h, rcond=None)[0]
            
            def obj_and_grad(coeffs):
                h = basis @ coeffs
                s = np.sum(h)
                if abs(s) < 1e-15:
                    return 1e10
                h = h / s
                conv = np.convolve(h, h)
                return abs(2 * N * np.max(conv))
            
            res = minimize(obj_and_grad, coeffs_init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                coeffs_fin = res.x
                h_fin = basis @ coeffs_fin
                s_fin = np.sum(h_fin)
                if abs(s_fin) > 1e-15:
                    h_fin = h_fin / s_fin
                    val_fin = _eval_value(h_fin)
                    if val_fin < best_val:
                        best_val = val_fin
                        best_h = h_fin.copy()
        except Exception:
            pass
    
    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-15:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
