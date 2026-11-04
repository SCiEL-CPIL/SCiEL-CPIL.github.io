"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def _get_linear_overlaps(h):
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

def _sturmian_sequence(alpha, beta, N):
    """
    Generate a Sturmian-like sequence based on fractional parts.
    h_i = 1 if {alpha * i + beta} < 0.5 else 0
    This creates a balanced binary sequence with good discrepancy properties.
    """
    idx = np.arange(N)
    frac = np.mod(alpha * idx + beta, 1.0)
    h = (frac < 0.5).astype(float)
    return h

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a uniform shift to achieve exact sum.
    """
    h = np.clip(h, 0, 1).copy()
    target = TARGET_SUM
    
    # Find the shift delta such that sum(clip(h + delta, 0, 1)) = target
    # Use bisection since sum(clip(h+delta,0,1)) is monotonically increasing in delta
    
    def _sum_at_delta(delta):
        return np.sum(np.clip(h + delta, 0, 1))
    
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        s = _sum_at_delta(mid)
        if s < target:
            lo = mid
        else:
            hi = mid
    
    delta = (lo + hi) / 2
    h_new = np.clip(h + delta, 0, 1)
    
    # Fine-tune: adjust one element to get exact sum
    sum_h = np.sum(h_new)
    diff = target - sum_h
    if abs(diff) > 1e-15:
        # Find an element that can absorb the difference
        if diff > 0:
            candidates = np.where(h_new < 1.0)[0]
            if len(candidates) > 0:
                idx = candidates[0]
                h_new[idx] = min(1.0, h_new[idx] + diff)
        else:
            candidates = np.where(h_new > 0.0)[0]
            if len(candidates) > 0:
                idx = candidates[0]
                h_new[idx] = max(0.0, h_new[idx] + diff)
    
    return h_new

def _objective_lse(x, beta, lam_sum):
    """
    Objective function using LogSumExp approximation of the max overlap.
    """
    h = np.clip(x, 0, 1)
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
        
    if max_ov > 1000:
        max_ov = 1000
        
    if beta > 0:
        shifted = overlaps - max_ov
        # Clip to prevent overflow in exp
        shifted = np.clip(shifted, -50/beta, 0)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, beta, lam_sum):
    """
    Objective function using epigraph formulation.
    x = [h_0, ..., h_{N-1}, t]
    Minimize t + sum_penalty
    Subject to: overlaps <= t (approximated via log-sum-exp)
    """
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        lse = 0.0
    else:
        max_ov = np.max(overlaps)
        if max_ov > 1000:
            max_ov = 1000
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50/beta, 0)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Penalize if lse > t
    if lse > t:
        penalty = (lse - t) ** 2
    else:
        penalty = 0.0
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return t + penalty + sum_penalty

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses epigraph formulation with auxiliary variable t for smoother optimization.
    Multiple diverse initializations including Sturmian sequences with adaptive beta schedule.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.default_rng(42)
    t_arr = np.arange(N)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sine wave at different frequencies
    for freq in [2, 3, 5, 7, 11]:
        h_init = np.full(N, 0.5)
        h_init += 0.3 * np.sin(2 * np.pi * freq * t_arr / N)
        initializations.append(h_init)
    
    # 2. Random perturbations
    for _ in range(5):
        h_init = np.full(N, 0.5)
        h_init += 0.2 * rng.standard_normal(N)
        initializations.append(h_init)
    
    # 3. Block patterns
    h_init = np.zeros(N)
    h_init[:N//2] = 1.0
    initializations.append(h_init)
    
    h_init = np.zeros(N)
    h_init[::2] = 1.0
    initializations.append(h_init)
    
    # 4. Sturmian-like sequences (quasi-periodic binary sequences)
    # Try several irrational slopes near golden ratio conjugate and others
    golden = (np.sqrt(5) - 1) / 2.0  # ~0.618
    for alpha in [golden, 0.7, 0.3, 0.4, 0.6, 0.8, 0.25, 0.75]:
        for beta in [0.0, 0.25, 0.5, 0.75]:
            h_sturm = _sturmian_sequence(alpha, beta, N)
            initializations.append(h_sturm)
    
    # 5. Smooth quasi-periodic (continuous version)
    for alpha in [golden, 0.7, 0.3]:
        for beta in [0.0, 0.5]:
            h_qp = np.mod(alpha * t_arr + beta, 1.0)
            h_qp = 0.5 + 0.5 * np.cos(2 * np.pi * h_qp)  # Smooth version
            initializations.append(h_qp)
    
    # Project all to feasible set and evaluate
    best_h = None
    best_score = np.inf
    t_init = 0.5
    
    for h_init in initializations:
        h_proj = _project_to_feasible(h_init)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
            t_init = score + 0.1
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_t = [(0.0, 10.0)]
    bounds = bounds_h + bounds_t
    
    # Phase 1: Progressive beta optimization
    beta_schedule = [10, 30, 100, 300, 1000, 3000]
    
    for beta in beta_schedule:
        if time.time() > deadline - 25.0:
            break
        
        x_init = np.concatenate([best_h, [t_init]])
        
        def obj_func(x):
            return _objective_epigraph(x, beta, 50.0)
        
        try:
            result = minimize(
                obj_func, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                t_init = max(0.0, result.x[N])
        except Exception:
            pass
        
        # Periodic restart from best with small perturbation
        if time.time() > deadline - 35.0:
            break
        
        h_perturbed = best_h + 0.05 * rng.standard_normal(N)
        h_perturbed = _project_to_feasible(h_perturbed)
        score_perturbed = _max_overlap(h_perturbed)
        
        if score_perturbed < best_score:
            best_score = score_perturbed
            best_h = h_perturbed
            t_init = best_score + 0.1
    
    # Phase 2: High-precision polish
    if time.time() < deadline - 20.0:
        x_init = np.concatenate([best_h, [t_init]])
        
        def obj_polish(x):
            return _objective_epigraph(x, 8000.0, 8000.0)
        
        try:
            result = minimize(
                obj_polish, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 60}
            )
            h_polished = _project_to_feasible(result.x[:N])
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Phase 3: Local coordinate descent on active overlaps
    if time.time() < deadline - 15.0:
        for _ in range(5):
            if time.time() > deadline - 15.0:
                break
            
            h_cd = best_h.copy()
            improved = True
            max_iter = 300
            
            for it in range(max_iter):
                if not improved:
                    break
                improved = False
                
                # Try small perturbations on each coordinate
                for i in range(N):
                    if time.time() > deadline - 15.0:
                        break
                    
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                        h_try = h_cd.copy()
                        h_try[i] = np.clip(h_try[i] + delta, 0, 1)
                        h_try = _project_to_feasible(h_try)
                        score_try = _max_overlap(h_try)
                        
                        if score_try < best_score - 1e-10:
                            best_score = score_try
                            best_h = h_try
                            h_cd = h_try.copy()
                            improved = True
                            break
    
    # Phase 4: Final fine-tuning with very small steps
    if time.time() < deadline - 5.0:
        h_final = best_h.copy()
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                for delta in [0.001, -0.001, 0.0005, -0.0005]:
                    h_try = h_final.copy()
                    h_try[i] = np.clip(h_try[i] + delta, 0, 1)
                    h_try = _project_to_feasible(h_try)
                    score_try = _max_overlap(h_try)
                    if score_try < best_score - 1e-12:
                        best_score = score_try
                        h_final = h_try.copy()
                        improved = True
                        break
            if not improved:
                break
        best_h = h_final
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    # Final precision: ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-15:
        if best_h[0] + diff <= 1.0 and best_h[0] + diff >= 0.0:
            best_h[0] += diff
        else:
            best_h = best_h * (TARGET_SUM / sum_h)
            best_h = np.clip(best_h, 0.0, 1.0)
            best_h = _project_to_feasible(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
