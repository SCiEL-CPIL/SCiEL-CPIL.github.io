"""
Strategy: Convex-Concave Fractional Program via Log-Height Reparameterization
and L-BFGS-B Optimization.

The objective value = l2 / (norm_1 * norm_inf) is scale-invariant.
We fix the scale by normalizing h such that sum(h) = 1 (or similar) to remove
one degree of freedom, or more effectively, we optimize over log(h) + epsilon
to ensure h > 0, and use L-BFGS-B which handles smooth optimization better than
coordinate descent in high dimensions (50D).

We define a smoothed objective that penalizes deviations from the target
norms or simply optimizes the ratio directly on the log-scale, where the
landscape is smoother.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # length of g = convolve(h, h)


def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    if np.all(h == 0):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    # l2 calculation
    # sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # y has length m+2. Indices 0..m+1.
    # The sum is over i=0 to m.
    # y[i] and y[i+1] are involved.
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def objective(log_h, eps=1e-8):
    """
    Objective function for minimization (negative value).
    log_h: vector of length N.
    h = exp(log_h)
    """
    h = np.exp(log_h)
    # Ensure finite
    if not np.all(np.isfinite(h)):
        return 1e10
    val = compute_value(h)
    return -val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    # Initial best
    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Generate diverse starting points for log_h
    # We want to explore different shapes.
    # h = exp(log_h).
    # Uniform h=1 -> log_h=0.
    
    x = np.linspace(-1, 1, N)
    
    starts_log_h = []
    
    # 1. Zero vector (uniform h)
    starts_log_h.append(np.zeros(N))
    
    # 2. Gaussian bumps in log-space (which means multiplicative bumps in h)
    # h = exp(-(x-c)^2 / 2s^2) -> log_h = -(x-c)^2 / 2s^2
    for c in np.linspace(-0.7, 0.7, 5):
        for s in np.linspace(0.2, 0.8, 5):
            log_h = -((x - c) ** 2) / (2 * s ** 2)
            starts_log_h.append(log_h)
            
    # 3. Linear ramps
    for slope in np.linspace(-2, 2, 5):
        log_h = slope * x
        starts_log_h.append(log_h)
        
    # 4. Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        log_h = rng.standard_normal(N) * 0.5
        starts_log_h.append(log_h)

    # Evaluate starts to find a good initial guess
    best_log_h = np.zeros(N)
    best_val_start = -1e10
    
    for log_h0 in starts_log_h:
        if time.time() > deadline - 20:
            break
        val = compute_value(np.exp(log_h0))
        if val > best_val_start:
            best_val_start = val
            best_log_h = log_h0.copy()
            best_h = np.exp(log_h0).copy()
            best_val = val

    # Local optimization using L-BFGS-B on log_h
    # We will run L-BFGS-B from the best start and a few other top starts.
    
    # Re-evaluate starts to get top 5
    start_vals = []
    for log_h0 in starts_log_h:
        if time.time() > deadline - 15:
            break
        val = compute_value(np.exp(log_h0))
        start_vals.append((val, log_h0))
    start_vals.sort(key=lambda x: x[0], reverse=True)
    
    top_starts = [lh for v, lh in start_vals[:5]]
    
    # Define bounds for log_h to prevent explosion or underflow
    # h can be very large or small. 
    # log_h range: [-10, 10] -> h in [4.5e-5, 22026]
    # This should be sufficient given scale invariance, but we might need wider.
    # Actually, since it's scale invariant, the optimizer might drift.
    # We can fix the scale by adding a constraint or just letting it drift.
    # L-BFGS-B handles bounds.
    bounds = [(-10, 10)] * N
    
    for log_h0 in top_starts:
        if time.time() > deadline - 5:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                objective, 
                log_h0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < -best_val:
                h_opt = np.exp(res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # If we have time, do a second pass with finer starts around the best solution
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        # Perturb best_h in log space
        log_h_best = np.log(best_h + 1e-10)
        
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # Small perturbation
            pert = rng.standard_normal(N) * 0.1
            log_h_try = log_h_best + pert
            # Clip to bounds
            log_h_try = np.clip(log_h_try, -10, 10)
            
            try:
                res = minimize(
                    objective, 
                    log_h_try, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < -best_val:
                    h_opt = np.exp(res.x)
                    val_opt = compute_value(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print(compute_value(h))
