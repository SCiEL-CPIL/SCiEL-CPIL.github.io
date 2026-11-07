"""Improved program for Erdos minimum overlap task using bisection on the maximum correlation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_correlations(h):
    """Compute the full correlation of h with 1-h."""
    return np.correlate(h, 1.0 - h, mode="full")

def objective_value(h):
    """Compute the task value."""
    c = compute_correlations(h)
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final adjustment for precision
    s = h.sum()
    if abs(s - target) > 1e-12:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def is_feasible(C, h_init=None, max_iter=200):
    """
    Check if there exists h such that max(c_k) <= C.
    Minimizes sum of squared violations of c_k <= C.
    """
    if h_init is None:
        h_init = np.full(N, 0.5)
    
    def violation_obj(h):
        h_proj = project(h)
        c = compute_correlations(h_proj)
        # Violation is where c_k > C
        viol = np.maximum(0.0, c - C)
        return float(np.sum(viol**2))
    
    def violation_grad(h):
        # Gradient of sum(max(0, c_k - C)^2)
        # d/dh sum(viol_k^2) = sum(2*viol_k * d(c_k)/dh)
        # This is expensive to compute analytically.
        # We will use numerical gradient or just use SLSQP which can handle it, 
        # but SLSQP needs constraints.
        # Let's use a different approach: Minimize a smooth surrogate of max(c_k) 
        # and check if it's <= C.
        pass

    # Alternative: Use SLSQP to minimize 0 subject to constraints c_k(h) <= C, 0<=h<=1, sum(h)=N/2
    # If SLSQP finds a solution, it's feasible.
    
    # To speed up, we can use the smooth max surrogate as the objective in a minimization,
    # and check if the minimum value is <= C.
    
    beta = 50.0
    
    def smooth_max_obj(h):
        h_proj = project(h)
        c = compute_correlations(h_proj)
        c_max = np.max(c)
        # Log-sum-exp for numerical stability
        shifted = beta * (c - c_max)
        log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
        return float(log_sum / N * 2.0) # This is V. We want V <= C/N*2? 
        # Wait, C in is_feasible is the max correlation value, not V.
        # Let's define C_corr = N * C_val / 2.
        # So we check if min smooth_max_corr <= C_corr.
        
    # Let's stick to the violation objective, it's more direct for bisection.
    # But we need gradients for L-BFGS-B.
    # Let's use SLSQP with constraints.
    
    constraints = []
    # Sum constraint
    constraints.append({'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)})
    
    # Correlation constraints: c_k(h) <= C
    # c_k(h) = sum_j h_j (1 - h_{j+k})
    # This is quadratic.
    # We can define the constraint function for each k.
    # 199 constraints might be heavy for SLSQP.
    
    # Better: Minimize the smooth max of c_k.
    # If min(smooth_max_c) <= C + epsilon, then feasible.
    
    def obj_smooth_max(h):
        h_proj = project(h)
        c = compute_correlations(h_proj)
        # Use beta to approximate max
        beta = 100.0
        c_max = np.max(c)
        # Avoid overflow
        shifted = beta * (c - c_max)
        log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
        return float(log_sum)
        
    def grad_smooth_max(h):
        # Numerical gradient is slow.
        # Let's just use SLSQP with finite diff if necessary, or L-BFGS-B with numerical grad.
        # L-BFGS-B is faster.
        return None

    # Let's try L-BFGS-B on the smooth max objective.
    # It doesn't support constraints, but project() handles sum and box.
    
    try:
        res = minimize(
            obj_smooth_max, 
            h_init, 
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        h_opt = project(res.x)
        c_opt = compute_correlations(h_opt)
        max_c = np.max(c_opt)
        return max_c <= C + 1e-6, h_opt, max_c
    except Exception:
        return False, h_init, float('inf')

def construct():
    t0 = time.time()
    
    # Initial guess
    best_h = np.full(N, 0.5)
    best_val = objective_value(best_h)
    
    # Bisection search for the minimum feasible max correlation C
    # V = 2/N * C_max
    # We want to minimize V, so minimize C_max.
    
    # Lower bound: Theoretical minimum? 
    # For random h, C_max is around 50.
    # We found ~38.2 (V~0.382).
    # Let's set lo = 30.0, hi = 50.0.
    
    lo = 30.0
    hi = 50.0
    
    # First, refine hi to be a known feasible point
    # Start with best_h from seed
    c_seed = compute_correlations(best_h)
    hi = np.max(c_seed)
    
    # Try to find a better starting point for the bisection
    # Use a few random restarts to find a good feasible C
    
    current_h = best_h.copy()
    
    # Run a few iterations of local search to improve current_h
    for _ in range(5):
        if remaining() < 20: break
        # Random perturbation
        noise = np.random.randn(N) * 0.1
        h_trial = project(current_h + noise)
        val_trial = objective_value(h_trial)
        if val_trial < best_val:
            best_h = h_trial
            best_val = val_trial
            current_h = h_trial
            hi = np.max(compute_correlations(h_trial))
    
    # Now bisect
    # We want to find the smallest C such that is_feasible(C) is True.
    # is_feasible uses optimization, which is expensive.
    # We will do ~15-20 bisection steps.
    
    num_steps = 20
    for i in range(num_steps):
        if remaining() < 10:
            break
            
        mid = (lo + hi) / 2.0
        
        feasible, h_found, c_found = is_feasible(mid, h_init=current_h)
        
        if feasible:
            hi = c_found # Use the actual found max for tighter bound
            if c_found < np.max(compute_correlations(best_h)):
                best_h = h_found
                best_val = objective_value(best_h)
                current_h = h_found
        else:
            lo = mid
            
        # Update best if we found something better in the process
        # (is_feasible returns the h that minimizes the smooth max, which might be good)
        if c_found < np.max(compute_correlations(best_h)):
             best_h = h_found
             best_val = objective_value(best_h)
             current_h = h_found

    # Final polish: Use the best_h found and run a more precise local optimization
    # Minimize the exact max (using high beta)
    
    if remaining() > 5:
        beta = 200.0
        def obj_exact_smooth(h):
            h_proj = project(h)
            c = compute_correlations(h_proj)
            c_max = np.max(c)
            shifted = beta * (c - c_max)
            log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
            return float(log_sum)
            
        try:
            res = minimize(
                obj_exact_smooth, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_polished = project(res.x)
            val_polished = objective_value(h_polished)
            if val_polished < best_val:
                best_h = h_polished
                best_val = val_polished
        except Exception:
            pass

    # Final projection and validation
    best_h = project(best_h)
    
    # Ensure sum is exactly N/2 within tolerance
    s = best_h.sum()
    if abs(s - N/2) > 1e-9:
        # Force correction
        diff = N/2 - s
        # Add diff to the element with the most "room"
        # Or just distribute
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
            # Re-project to fix sum again if clipping broke it
            best_h = project(best_h)

    return best_h
