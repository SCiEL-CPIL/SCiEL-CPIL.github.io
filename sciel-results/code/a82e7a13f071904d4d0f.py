"""Rewritten program for second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N:
        return 0.0
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < 0):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # spacing w
    # Points are np.linspace(-0.5, 0.5, m+2)
    # Number of intervals = m+1
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Note: i ranges from 0 to m (inclusive), so m+1 terms
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    
    # Seed value
    seed_h = np.ones(N)
    seed_value = compute_value(seed_h)
    
    best_h = seed_h.copy()
    best_value = seed_value
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Strategy: The objective is scale-invariant in a certain sense.
    # Let's analyze the structure.
    # g = h * h (convolution)
    # value = l2 / (norm_1 * norm_inf)
    
    # Let's try to optimize using gradient-based methods from multiple starts.
    # Since h >= 0, we can parameterize h = exp(x) or use bounds [0, max_h].
    # Note that scaling h by c scales g by c^2, l2 by c^4, norm_1 by c^2, norm_inf by c^2.
    # So value is scale-invariant! value(c*h) = value(h).
    # This means we can fix one degree of freedom, e.g., sum(h) = 1 or max(h) = 1.
    # Let's fix sum(h) = 1 for optimization, then the actual h doesn't matter for value.
    
    # We'll optimize over h with sum(h)=1, h>=0.
    # Using log-barrier or simple projection.
    # Actually, simpler: optimize over x where h_i = exp(x_i) / sum(exp(x_j)).
    # Or just use scipy minimize with bounds and constraint sum(h)=1.
    
    # Let's try a few random starts with L-BFGS-B or Powell.
    # Powell doesn't require gradients and handles bounds well.
    
    def neg_value(h):
        return -compute_value(h)
    
    # Constraint: sum(h) = 1
    # We'll use a penalty method or just rely on the fact that value is scale invariant.
    # If we don't constrain sum, the optimizer might drift to very small or large values.
    # But since it's scale invariant, we can just add a small penalty for sum(h) != 1.
    
    def objective_with_penalty(h):
        val = compute_value(h)
        penalty = 0.1 * (np.sum(h) - 1.0)**2
        return -val + penalty
    
    # Bounds: h_i in [0, 1] (since sum=1, max is 1)
    bounds = [(0.0, 1.0)] * N
    
    # Try a few random initial points
    rng = np.random.RandomState(42)
    
    num_starts = 20
    for i in range(num_starts):
        if elapsed() > DEADLINE - 10:
            break
        
        # Random initialization
        h0 = rng.dirichlet(np.ones(N))
        
        # Use Powell's method
        try:
            res = minimize(
                objective_with_penalty,
                h0,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            
            h_opt = res.x
            val_opt = compute_value(h_opt)
            
            if val_opt > best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Also try some structured initializations
    # 1. Uniform (already seed)
    # 2. Triangular
    # 3. Gaussian-like
    # 4. Concentrated in middle
    # 5. Concentrated at edges
    
    structured_starts = []
    
    # Triangular
    h_tri = np.linspace(0, 1, N)
    h_tri = h_tri / np.sum(h_tri)
    structured_starts.append(h_tri)
    
    # Inverted triangular
    h_inv_tri = np.linspace(1, 0, N)
    h_inv_tri = h_inv_tri / np.sum(h_inv_tri)
    structured_starts.append(h_inv_tri)
    
    # Gaussian centered
    x = np.arange(N)
    h_gauss = np.exp(-((x - N/2.0)**2) / (2 * (N/6.0)**2))
    h_gauss = h_gauss / np.sum(h_gauss)
    structured_starts.append(h_gauss)
    
    # Concentrated in middle
    h_mid = np.zeros(N)
    h_mid[N//2 - 5:N//2 + 5] = 1.0
    h_mid = h_mid / np.sum(h_mid)
    structured_starts.append(h_mid)
    
    # Concentrated at edges
    h_edge = np.zeros(N)
    h_edge[:5] = 1.0
    h_edge[-5:] = 1.0
    h_edge = h_edge / np.sum(h_edge)
    structured_starts.append(h_edge)
    
    # Bimodal
    h_bi = np.zeros(N)
    h_bi[N//4 - 3:N//4 + 3] = 1.0
    h_bi[3*N//4 - 3:3*N//4 + 3] = 1.0
    h_bi = h_bi / np.sum(h_bi)
    structured_starts.append(h_bi)
    
    for h0 in structured_starts:
        if elapsed() > DEADLINE - 10:
            break
        
        try:
            res = minimize(
                objective_with_penalty,
                h0,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            
            h_opt = res.x
            val_opt = compute_value(h_opt)
            
            if val_opt > best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Try differential evolution if time permits
    if elapsed() < DEADLINE - 30:
        try:
            def de_objective(h):
                return -compute_value(h)
            
            # DE is global optimizer, good for non-convex problems
            res_de = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                init='sobol'
            )
            
            h_de = res_de.x
            val_de = compute_value(h_de)
            
            if val_de > best_value:
                best_value = val_de
                best_h = h_de.copy()
                
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead on best solution found
    if elapsed() < DEADLINE - 5:
        try:
            res_final = minimize(
                objective_with_penalty,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            h_final = res_final.x
            # Clip to valid range
            h_final = np.clip(h_final, 0, 1)
            val_final = compute_value(h_final)
            
            if val_final > best_value:
                best_value = val_final
                best_h = h_final.copy()
                
        except Exception:
            pass
    
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = seed_h
    
    return list(best_h)
