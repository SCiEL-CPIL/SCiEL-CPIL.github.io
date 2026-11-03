"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, gamma
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # P(x) = prod (x - x_i)^2
    # C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    
    # Use Gauss-Laguerre quadrature
    # The integrals are over [0, infinity) with weight e^{-x}
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) at quadrature points
    # P(x) = prod (x - x_i)^2
    # Use log-sum-exp style stability if possible, but with n=12, direct is usually fine
    # unless roots are very large.
    
    log_P = np.zeros(n_quad)
    for r in roots:
        # Avoid log(0) if x_quad hits a root exactly (unlikely with Gauss nodes)
        # But roots are double, so P(x) >= 0.
        # P(x) = prod (x-r)^2.
        # log P = sum 2 log |x-r|
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
        
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective_geometric(params):
    """
    Objective function for geometric parameterization.
    params: [scale]
    Roots are scale * laguerre_roots.
    """
    scale = params[0]
    if scale <= 0:
        return 1e100
    x0, _ = roots_laguerre(12)
    roots = scale * x0
    return _compute_c4_bound(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin
    
    # Strategy: 
    # 1. The optimal roots for Cohn-Goncalves are often close to scaled Laguerre roots.
    # 2. We can parameterize roots as s * L_roots.
    # 3. Optimize s.
    # 4. Then, refine with local optimization on the full 12D space starting from best geometric solution.
    
    # Initial guess: Standard Laguerre roots correspond to s=1
    x0, _ = roots_laguerre(12)
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Phase 1: Optimize the scale factor s
    # s * x0
    # The optimal s might be around 1.0 but could vary.
    # Let's do a coarse grid search first to be safe, then Nelder-Mead.
    
    # Grid search for s in [0.1, 5.0]
    s_grid = np.linspace(0.1, 5.0, 50)
    best_s = 1.0
    for s in s_grid:
        if time.time() > deadline - 100:
            break
        roots = s * x0
        score = _compute_c4_bound(roots)
        if score < best_score:
            best_score = score
            best_roots = roots.copy()
            best_s = s
            
    # Nelder-Mead on s around best_s
    if time.time() < deadline - 80:
        try:
            res = minimize(
                _objective_geometric,
                [best_s],
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000}
            )
            if res.fun < best_score:
                best_s = res.x[0]
                best_roots = best_s * x0
                best_score = res.fun
        except Exception:
            pass

    # Phase 2: Local refinement in 12D space
    # Start from the geometric solution
    # Use L-BFGS-B with bounds
    bounds = [(1e-6, None)] * 12
    
    if time.time() < deadline - 30:
        try:
            # We need a wrapper that takes 12D vector
            def _objective_12d(roots):
                return _compute_c4_bound(roots)
            
            result = minimize(
                _objective_12d,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                candidate = result.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
            
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
