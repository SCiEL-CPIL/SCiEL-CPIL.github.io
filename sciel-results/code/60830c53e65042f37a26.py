import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp

N = 100
DEADLINE = 110.0

def get_correlations(h):
    """Compute the correlation values C_k."""
    # np.correlate(h, 1-h, mode="full")
    # This computes sum_i h_i * (1-h)_{i+k}
    # Note: np.correlate(h, v) computes sum_i h_{i+k} v_i.
    # So np.correlate(h, 1-h) computes sum_i h_{i+k} (1-h)_i.
    # This is the same as sum_i h_i (1-h)_{i-k} by symmetry of correlation?
    # Actually, correlate(h, v) = sum h_{i+k} v_i.
    # We want sum h_i (1-h)_{i+k}.
    # Let v = 1-h.
    # correlate(h, v) gives sum h_{i+k} v_i.
    # This is the correlation of h with v, lagged by k.
    # The set of values is the same for k and -k.
    # So max is the same.
    return np.correlate(h, 1 - h, mode="full")

def objective_exact(h):
    corr = get_correlations(h)
    return np.max(corr)

def objective_smooth(h, beta=100.0):
    corr = get_correlations(h)
    # Scale for stability
    m = np.max(corr)
    if m == 0: return 0.0
    # logsumexp(corr * beta) / beta approximates max(corr)
    # But if corr has large negative values, exp(beta * corr) might underflow.
    # logsumexp handles this by subtracting max.
    return logsumexp(corr * beta) / beta

def project_to_simplex(h, target_sum=N/2):
    """Project h to [0,1]^N with sum h = target_sum."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    if abs(s - target_sum) < 1e-12:
        return h
    
    # Water filling / iterative projection
    for _ in range(100):
        at_lower = h <= 1e-12
        at_upper = h >= 1 - 1e-12
        free = ~(at_lower | at_upper)
        
        if not np.any(free):
            break
            
        diff = target_sum - np.sum(h)
        n_free = np.sum(free)
        h[free] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
        
    # Final adjustment
    s = np.sum(h)
    if abs(s - target_sum) > 1e-12:
        diff = target_sum - s
        free = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
            
    return h

def construct():
    start_time = time.time()
    
    # Initial candidates
    h_seed = np.full(N, 0.5)
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    
    best_h = h_alt.copy()
    best_val = objective_exact(best_h)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return best_h

    # Differential Evolution
    # We minimize objective_exact + penalty for sum constraint
    penalty_weight = 100.0
    
    def de_objective(h):
        val = objective_exact(h)
        penalty = penalty_weight * (np.sum(h) - N/2)**2
        return val + penalty

    bounds = [(0, 1)] * N
    
    try:
        # DE is stochastic, so we use a fixed seed for reproducibility in this context
        # but the task implies we just need to find a good solution.
        # maxiter=100 might be slow. Let's try 50.
        # popsize=10 (default is 15).
        res_de = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-5,
            seed=42,
            polish=False # We will polish manually
        )
        
        if res_de.success or res_de.fun < best_val:
            h_de = res_de.x
            h_de = project_to_simplex(h_de)
            v_de = objective_exact(h_de)
            if v_de < best_val:
                best_h = h_de
                best_val = v_de
    except Exception:
        pass

    if time.time() - start_time > DEADLINE - 10:
        return best_h

    # Polish with SLSQP
    # Use smooth objective for gradients
    beta = 100.0
    
    def slsqp_objective(h):
        return objective_smooth(h, beta)

    def slsqp_jacobian(h):
        # Gradient of logsumexp(beta * corr) / beta
        # d/dh logsumexp(x) = softmax(x)
        # x = beta * corr
        # corr is linear in h? No, corr is quadratic.
        # C_k = sum h_i (1-h_{i+k})
        # d C_k / d h_j = (1-h_{j+k}) - h_j (if j+k in range) ...
        # This is complex to derive analytically for the full max.
        # Let's let SLSQP use finite differences or provide a numerical gradient.
        # Or just use 'BFGS' or 'Nelder-Mead' if SLSQP struggles with gradients?
        # SLSQP requires a gradient or uses finite differences.
        # Let's try providing the gradient numerically or just letting it estimate.
        # Actually, `minimize` with method='SLSQP' and `jac=None` uses finite differences.
        # For N=100, this is 100 evaluations per gradient step. Might be slow.
        # Let's try `L-BFGS-B` with a projection trick?
        # No, L-BFGS-B can't do equality constraints.
        
        # Alternative: Minimize `objective_smooth` with `method='SLSQP'`.
        # To speed up, we can reduce beta or use a coarser polish.
        
        return slsqp_objective(h)

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}
    ]
    
    try:
        res_slsqp = minimize(
            slsqp_objective,
            best_h,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res_slsqp.success:
            h_slsqp = res_slsqp.x
            h_slsqp = project_to_simplex(h_slsqp)
            v_slsqp = objective_exact(h_slsqp)
            if v_slsqp < best_val:
                best_h = h_slsqp
                best_val = v_slsqp
    except Exception:
        pass

    if time.time() - start_time > DEADLINE - 5:
        return best_h

    # Try a few random restarts with DE if time remains
    rng = np.random.default_rng(123)
    
    for i in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Random start
        h_rand = rng.random(N)
        h_rand = project_to_simplex(h_rand)
        
        # Run a short DE or just SLSQP from this point?
        # SLSQP is local. DE is global.
        # Let's just run SLSQP from random starts to see if we find a better local min.
        
        try:
            res = minimize(
                slsqp_objective,
                h_rand,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.success:
                h_new = project_to_simplex(res.x)
                v_new = objective_exact(h_new)
                if v_new < best_val:
                    best_h = h_new
                    best_val = v_new
        except Exception:
            pass

    # Final check and projection
    best_h = project_to_simplex(best_h)
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.any(best_h < -1e-9) or np.any(best_h > 1 + 1e-9):
        best_h = h_alt
    if abs(np.sum(best_h) - N/2) > 1e-9 * N:
        best_h = h_alt

    return best_h
