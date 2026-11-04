"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k != 0.
    C_k = sum_i h_i * (1 - h_{i+k})
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses exact convex projection via bisection on Lagrange multiplier.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM
    
    current = np.sum(h)
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection to find shift that makes sum = target
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = np.sum(np.clip(h + mid, 0.0, 1.0))
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        free = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_epigraph(x, t, beta, lam_sum):
    """
    Epigraph formulation: minimize t + penalty subject to all overlaps <= t.
    Uses smooth approximation of the constraint violations.
    """
    h = np.clip(x, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        violation = 0.0
    else:
        # Smooth approximation of max(0, overlaps - t)
        shifted = overlaps - t
        violation = np.sum(np.log1p(np.exp(beta * shifted))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return t + violation + sum_penalty

def _generate_h_from_theta(theta):
    """
    Generate h from a richer parameterization.
    theta = [f1, f2, f3, a1, a2, a3, c0]
    where h is a sum of sinusoids with frequencies f1,f2,f3 and amplitudes a1,a2,a3.
    """
    f1, f2, f3 = theta[0], theta[1], theta[2]
    a1, a2, a3 = theta[3], theta[4], theta[5]
    c0 = theta[6]
    
    n = np.arange(N)
    h = 0.5
    h += a1 * np.sin(2 * np.pi * f1 * n / N)
    h += a2 * np.sin(2 * np.pi * f2 * n / N)
    h += a3 * np.sin(2 * np.pi * f3 * n / N)
    
    # Scale by c0
    h = 0.5 + c0 * (h - 0.5)
    
    h = _project_to_feasible(h)
    return h

def _objective_theta(theta):
    """
    Objective for sinusoid parameter optimization.
    """
    h = _generate_h_from_theta(theta)
    return _max_overlap(h)

def _coordinate_descent_polish(h, max_iter=10000):
    """
    Coordinate descent polish: iteratively adjust each h_i to reduce max overlap.
    """
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    overlaps = _compute_linear_overlaps(best_h)
    current_max = np.max(overlaps) if len(overlaps) > 0 else 0.0
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            # Try small perturbations
            for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                h_new = best_h.copy()
                h_new[i] = np.clip(h_new[i] + delta, 0.0, 1.0)
                
                # Adjust sum to maintain constraint
                sum_diff = TARGET_SUM - np.sum(h_new)
                if abs(sum_diff) > 1e-14:
                    # Distribute among other non-saturated elements
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    free = mask & (h_new > 1e-10) & (h_new < 1.0 - 1e-10)
                    if np.any(free):
                        h_new[free] += sum_diff / np.sum(free)
                        h_new = np.clip(h_new, 0.0, 1.0)
                
                score_new = _max_overlap(h_new)
                if score_new < best_score - 1e-12:
                    best_h = h_new
                    best_score = score_new
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_h

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Strategy:
    1. Low-dimensional search over sinusoid parameters.
    2. Full-dimensional L-BFGS-B refinement using epigraph formulation.
    3. Coordinate descent polish for high precision.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h_low_dim = None
    best_score_low_dim = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Parameters: [f1, f2, f3, a1, a2, a3, c0]
    # Frequencies in [1, 20], amplitudes in [0.05, 0.3], c0 in [0.3, 0.8]
    bounds_theta = [
        (1, 20), (1, 20), (1, 20),
        (0.05, 0.3), (0.05, 0.3), (0.05, 0.3),
        (0.3, 0.8)
    ]
    
    # Try multiple random starts for the low-dim optimization
    num_starts = 25
    for i in range(num_starts):
        if time.time() > deadline - 40.0:
            break
            
        # Random start with some structured initialization
        if i < 5:
            # Structured starts: try different frequency combinations
            f1 = rng.choice([3, 5, 7, 10, 13])
            f2 = rng.choice([2, 4, 6, 8, 11, 15])
            f3 = rng.choice([1, 3, 5, 7, 9, 12, 17])
            theta0 = np.array([f1, f2, f3, 0.15, 0.1, 0.05, 0.5])
        else:
            theta0 = np.array([
                rng.uniform(1, 20), rng.uniform(1, 20), rng.uniform(1, 20),
                rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3),
                rng.uniform(0.3, 0.8)
            ])
        
        try:
            res = minimize(
                _objective_theta, 
                theta0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            h_cand = _generate_h_from_theta(res.x)
            score_cand = _max_overlap(h_cand)
            
            if score_cand < best_score_low_dim:
                best_score_low_dim = score_cand
                best_h_low_dim = h_cand
        except Exception:
            pass

    # If low-dim search failed, fallback to simple sinusoid
    if best_h_low_dim is None:
        best_h_low_dim = _project_to_feasible(0.5 + 0.3 * np.sin(2 * np.pi * 5 * np.arange(N) / N))
        best_score_low_dim = _max_overlap(best_h_low_dim)

    # 2. Full-dimensional refinement using epigraph formulation
    best_h = best_h_low_dim.copy()
    best_score = best_score_low_dim
    
    # Use a fixed t initialized to current max overlap
    t_init = best_score + 0.01
    
    # Create combined parameter vector [h, t]
    bounds_combined = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    def obj_combined(xt):
        x = xt[:N]
        t = xt[N]
        return _objective_epigraph(x, t, 50.0, 100.0)
    
    xt_init = np.concatenate([best_h, [t_init]])
    
    # Refinement with increasing beta
    beta_schedule = [20, 50, 100, 200]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
        
        def obj_func(xt):
            x = xt[:N]
            t = xt[N]
            return _objective_epigraph(x, t, beta, 100.0)
        
        try:
            result = minimize(
                obj_func, xt_init, method='L-BFGS-B', bounds=bounds_combined,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                xt_init = np.concatenate([best_h, [score_new + 0.01]])
        except Exception:
            pass

    # 3. Coordinate descent polish
    if time.time() < deadline - 5.0:
        best_h = _coordinate_descent_polish(best_h, max_iter=5000)
        best_score = _max_overlap(best_h)
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    # Final precision polish: ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        free = (best_h > 1e-10) & (best_h < 1.0 - 1e-10)
        if np.any(free):
            best_h[free] += diff / np.sum(free)
            best_h = np.clip(best_h, 0.0, 1.0)
            best_h = _project_to_feasible(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
