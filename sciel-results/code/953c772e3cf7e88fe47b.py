"""Optimized program for third autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize, line_search
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill


def _evaluate(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _best_value_for_direction(v):
    """
    Given a zero-sum unit-norm direction v, find the best t = a/b
    to minimize the objective for h = a + b*v (with b=1 wlog).
    h = t + v, where v has zero mean.
    sum(h) = N*t
    conv(h,h) = conv(t*1 + v, t*1 + v) = t^2 * conv(1,1) + 2*t*conv(1,v) + conv(v,v)
    
    For the max of conv(h,h), we need to find t that minimizes
    |2*N*max(conv(t+v, t+v)) / (N*t)^2|
    
    Actually, let's just numerically optimize over t for a given v.
    """
    # h = t + v, sum(h) = N*t (since sum(v)=0)
    # We need sum(h) != 0, so t != 0
    # Objective: |2*N*max(convolve(t+v, t+v)) / (N*t)^2|
    
    def obj(t):
        if abs(t) < 1e-10:
            return np.inf
        h = t + v
        s = N * t  # since sum(v) = 0
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    # Try a range of t values
    best_val = np.inf
    best_t = 0.0
    
    # Try positive and negative t
    for t_init in [0.1, 0.5, 1.0, 2.0, 5.0, -0.1, -0.5, -1.0, -2.0, -5.0]:
        res = minimize(obj, t_init, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_t = res.x[0]
    
    return best_val, best_t


def _make_zero_sum_unit(v):
    """Make v zero-sum and unit norm."""
    v = v - np.mean(v)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(N)
    return v / n


def construct():
    start_time = time.time()
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # should be 2.0
    
    rng = np.random.RandomState(42)
    
    # Strategy: random zero-sum directions, then optimize t analytically/numerically
    # and use L-BFGS to refine the direction
    
    attempts = 0
    max_attempts = 200
    
    while time.time() - start_time < DEADLINE and attempts < max_attempts:
        attempts += 1
        
        # Generate random zero-sum direction
        v_raw = rng.randn(N)
        v = _make_zero_sum_unit(v_raw)
        
        # Find best t for this direction
        val, t_opt = _best_value_for_direction(v)
        
        if val < best_val:
            h_new = t_opt + v
            best_val = val
            best_h = h_new.copy()
        
        # Try L-BFGS refinement on the direction
        # Parameterize: h = t + v, optimize v (subject to sum(v)=0, ||v||=1) and t
        # Use a parameterization that automatically enforces constraints
        # v = normalize(x - mean(x)) where x is free
        
        def full_obj(params):
            x = params[:N]
            t = params[N]
            v_dir = x - np.mean(x)
            norm = np.linalg.norm(v_dir)
            if norm < 1e-12:
                return np.inf
            v_dir = v_dir / norm
            h = t + v_dir
            s = np.sum(h)
            if abs(s) < 1e-10:
                return np.inf
            conv = np.convolve(h, h)
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        if time.time() - start_time > DEADLINE:
            break
        
        # Initialize from current best or random
        if attempts % 5 == 0:
            # Use current best as starting point
            x_init = best_h - np.mean(best_h)
            norm_x = np.linalg.norm(x_init)
            if norm_x > 1e-12:
                x_init = x_init / norm_x
            t_init = np.mean(best_h)
        else:
            x_init = rng.randn(N)
            x_init = x_init - np.mean(x_init)
            norm_x = np.linalg.norm(x_init)
            if norm_x > 1e-12:
                x_init = x_init / norm_x
            t_init = rng.uniform(0.1, 2.0)
        
        params_init = np.concatenate([x_init, [t_init]])
        
        try:
            res = minimize(full_obj, params_init, method='L-BFGS-B',
                         options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                x_opt = res.x[:N]
                t_opt2 = res.x[N]
                v_opt = x_opt - np.mean(x_opt)
                norm_v = np.linalg.norm(v_opt)
                if norm_v > 1e-12:
                    v_opt = v_opt / norm_v
                    h_new = t_opt2 + v_opt
                    val_new = _evaluate(h_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_h = h_new.copy()
        except Exception:
            pass
    
    # Final check
    final_val = _evaluate(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h


# Quick test
if __name__ == '__main__':
    h = construct()
    val = _evaluate(h)
    print(f"Value: {val}")
    print(f"Score: {2.0 / val}")
