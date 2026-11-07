"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _evaluate(h):
    """Compute the objective value: max correlate / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _objective_from_control(control):
    """
    Map 10 control points to 100 heights, project to sum constraint,
    and return the objective value.
    """
    # Control points define values at indices 0, 10, 20, ..., 90 (10 points)
    # Linearly interpolate to 100 points
    ctrl_indices = np.linspace(0, 99, 10)
    h_raw = np.interp(np.arange(100), ctrl_indices, control)
    
    # Project to [0, 1] and enforce sum constraint
    # First clip to [0, 1]
    h = np.clip(h_raw, 0.0, 1.0)
    
    # Adjust to satisfy sum = 50 (len(h)/2 = 50)
    target_sum = N / 2.0
    current_sum = np.sum(h)
    
    if current_sum > 0:
        # Scale towards target, but need to handle bounds
        # Use a simple projection: adjust by adding/subtracting uniformly
        # then clip and iterate a few times
        for _ in range(20):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            # Adjust each element by diff/N, but respect bounds
            h_new = h + diff / N
            h_new = np.clip(h_new, 0.0, 1.0)
            # If clipping changed the sum, recompute
            if np.allclose(h_new, h, atol=1e-14):
                h = h_new
                break
            h = h_new
        
        # Final check: if sum is still off, do a more careful projection
        # Use a water-filling approach
        if abs(np.sum(h) - target_sum) > 1e-9:
            # Water filling to get exact sum while respecting [0,1]
            h = np.clip(h_raw, 0.0, 1.0)
            for _ in range(100):
                s = np.sum(h)
                if abs(s - target_sum) < 1e-12:
                    break
                delta = (target_sum - s) / N
                h = np.clip(h + delta, 0.0, 1.0)
    
    # Ensure finite
    h = np.where(np.isfinite(h), h, 0.5)
    
    return _evaluate(h)

def _construct_from_control(control):
    """Map control points to 100 heights with sum constraint."""
    ctrl_indices = np.linspace(0, 99, 10)
    h_raw = np.interp(np.arange(100), ctrl_indices, control)
    h = np.clip(h_raw, 0.0, 1.0)
    
    target_sum = N / 2.0
    for _ in range(100):
        s = np.sum(h)
        if abs(s - target_sum) < 1e-12:
            break
        delta = (target_sum - s) / N
        h = np.clip(h + delta, 0.0, 1.0)
    
    h = np.where(np.isfinite(h), h, 0.5)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with uniform 0.5
    best_control = np.full(10, 0.5)
    best_value = _evaluate(np.full(N, 0.5))
    
    # Try some initial diverse control point configurations
    initial_configs = [
        np.full(10, 0.5),
        np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5]),
        np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5]),
        np.array([1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5]),
    ]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        val = _objective_from_control(config)
        if val < best_value:
            best_value = val
            best_control = config.copy()
    
    # Use L-BFGS-B to optimize the 10 control points
    # We need a smooth objective. The max in correlate is nonsmooth,
    # but we can try direct optimization first, then polish.
    
    def obj_func(control):
        return _objective_from_control(control)
    
    # Multiple restarts with different initial points
    rng = np.random.RandomState(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_control.copy()
        elif restart == 1:
            x0 = np.full(10, 0.5) + rng.normal(0, 0.1, 10)
        else:
            # Random perturbation of best
            x0 = best_control + rng.normal(0, 0.2, 10)
        
        x0 = np.clip(x0, 0.0, 1.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * 10,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_control = result.x.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead as a local search on the 10D control space
    if time.time() < deadline:
        try:
            result = minimize(
                obj_func,
                best_control,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_control = result.x.copy()
        except Exception:
            pass
    
    # Final polish: try to optimize the full 100 heights locally
    # using the current best as initialization
    if time.time() < deadline - 10:
        try:
            h_best = _construct_from_control(best_control)
            
            def obj_full(h):
                h = np.clip(h, 0.0, 1.0)
                # Enforce sum constraint via penalty
                sum_penalty = (np.sum(h) - N/2.0) ** 2 * 1000
                corr = np.correlate(h, 1 - h, mode="full")
                return np.max(corr) / N * 2 + sum_penalty
            
            result = minimize(
                obj_full,
                h_best,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            h_opt = np.clip(result.x, 0.0, 1.0)
            # Project to exact sum
            for _ in range(100):
                s = np.sum(h_opt)
                if abs(s - N/2.0) < 1e-12:
                    break
                delta = (N/2.0 - s) / N
                h_opt = np.clip(h_opt + delta, 0.0, 1.0)
            
            val = _evaluate(h_opt)
            if val < best_value:
                best_value = val
                # Keep h_opt as final result
                best_h = h_opt
            else:
                best_h = h_best
        except Exception:
            best_h = _construct_from_control(best_control)
    else:
        best_h = _construct_from_control(best_control)
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    for _ in range(100):
        s = np.sum(best_h)
        if abs(s - N/2.0) < 1e-12:
            break
        delta = (N/2.0 - s) / N
        best_h = np.clip(best_h + delta, 0.0, 1.0)
    
    best_h = np.where(np.isfinite(best_h), best_h, 0.5)
    
    return best_h

# EVOLVE-BLOCK-END
