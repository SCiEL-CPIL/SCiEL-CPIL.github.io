"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre, roots_genlaguerre

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    DEADLINE = 585  # safety margin before 600s kill
    
    m = 12
    n = 1
    alpha = 2 * n  # = 2
    
    # Start from the zeros of L_12^(2)(x) - these gave score 0.486 in prior runs
    try:
        lag_roots, _ = roots_genlaguerre(m, alpha)
        x0 = np.sort(lag_roots).astype(float)
        # Ensure all positive
        if np.any(x0 <= 0):
            x0 = np.array([4.0 * (i + 1) for i in range(m)])
    except Exception:
        x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    def check_validity(x):
        """Check if roots are positive and distinct."""
        if np.any(x <= 0):
            return False
        # Check distinctness: minimum distance between any pair should be > threshold
        # Using sorting and diff is O(N log N)
        x_sorted = np.sort(x)
        min_dist = np.min(np.diff(x_sorted))
        return min_dist > 1e-6

    def objective(x):
        """
        Surrogate for the C4 upper bound.
        Includes strong penalties for invalid states (non-positive, non-distinct).
        """
        if np.any(x <= 0):
            return 1e10
            
        x_sorted = np.sort(x)
        min_dist = np.min(np.diff(x_sorted))
        
        # Penalty for non-distinct roots
        # If roots are too close, add a large penalty
        if min_dist < 1e-4:
            # Quadratic penalty that grows as they get closer
            distinct_penalty = (1e-4 - min_dist) ** 2 * 1e6
        else:
            distinct_penalty = 0.0

        # Primary term: largest root (dominates the bound)
        r_max = x_sorted[-1]
        
        # Secondary: evaluate L_m^(alpha) at the largest root
        # If L_m^(alpha)(r_max) is close to 0, the root is "exact"
        try:
            lag_val = eval_genlaguerre(m, alpha, r_max)
            # Penalize deviation from zero
            lag_penalty = abs(lag_val) * 0.1
        except Exception:
            lag_penalty = 0.0
        
        # Tertiary: spacing regularity (penalize irregular spacing)
        if len(x_sorted) > 1:
            spacings = np.diff(x_sorted)
            spacing_var = np.var(spacings) / (np.mean(spacings) ** 2 + 1e-10)
            spacing_penalty = spacing_var * 0.01
        else:
            spacing_penalty = 0.0
        
        return r_max + lag_penalty + spacing_penalty + distinct_penalty
    
    # Bounds: all roots must be positive
    bounds = [(1e-8, 100.0) for _ in range(m)]
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Local optimization from the Laguerre roots
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Try SLSQP as well
    try:
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Try Nelder-Mead for non-smooth exploration
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                     'adaptive': True}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - t0 > DEADLINE - 60:
        best_x = np.sort(np.maximum(best_x, 1e-8))
        return [float(r) for r in best_x]
    
    # Multi-start: perturb around best and re-optimize
    np.random.seed(42)
    for trial in range(10):
        if time.time() - t0 > DEADLINE - 30:
            break
        # Random perturbation
        scale = np.array([max(0.1, 0.05 * r) for r in best_x])
        perturbation = np.random.normal(0, 1.0, m) * scale
        x_trial = best_x + perturbation
        x_trial = np.maximum(x_trial, 1e-8)
        
        try:
            result = minimize(
                objective, x_trial, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Final polish: high-precision local optimization
    if time.time() - t0 < DEADLINE - 10:
        try:
            result = minimize(
                objective, best_x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Ensure sorted, positive
    best_x = np.sort(np.maximum(best_x, 1e-8))
    
    # Safety: if result is worse than Laguerre roots, fall back
    lag_obj = objective(x0)
    
    # Critical Fix: Ensure the final result is valid (distinct and positive)
    # If the optimized result is invalid, fall back to the initial Laguerre roots
    # which are guaranteed to be distinct and positive.
    if not check_validity(best_x):
        best_x = x0
    elif objective(best_x) > lag_obj * 1.01:
        best_x = x0
        
    # Final check on the chosen best_x
    if not check_validity(best_x):
        # This should never happen as x0 is valid, but just in case
        # Construct a simple valid set
        best_x = np.array([1.0 + i * 0.1 for i in range(m)])

    return [float(r) for r in best_x]
# EVOLVE-BLOCK-END
