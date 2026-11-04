"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _overlaps_from_h(h):
    """Compute circular overlap O_k = sum_i h[i]*(1-h[(i+k)%N]) for all k."""
    g = 1.0 - h
    # Circular correlation
    c = np.correlate(h, g, mode='full')
    c1 = c[:N]
    c2 = c[N+1:]
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _overlaps_from_h(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) < 1e-12:
        return h
    for _ in range(300):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            room = 1.0 - h
            room[h >= 1.0 - 1e-15] = 0
            total_room = room.sum()
            if total_room < 1e-15:
                break
            h += room * (diff / total_room)
            h = np.clip(h, 0, 1)
        else:
            avail = h
            avail[h <= 1e-15] = 0
            total_avail = avail.sum()
            if total_avail < 1e-15:
                break
            h -= avail * (-diff / total_avail)
            h = np.clip(h, 0, 1)
    return h

def _sturmian_heights(alpha, phase, N_local):
    """
    Generate step heights from a Sturmian-like sequence.
    h[i] = 0.5 + A * (2*frac((i+phase)*alpha) - 1)
    where A controls amplitude. We optimize alpha, phase, and A.
    """
    t = np.arange(N_local, dtype=np.float64)
    frac = np.mod((t + phase) * alpha, 1.0)
    # Map fractional part to [-1, 1]
    x = 2.0 * frac - 1.0
    # Use a smooth S-curve to get heights in [0,1]
    # h = 0.5 + 0.5 * tanh(k * x) gives values in (0,1)
    # But for better control, use a polynomial mapping
    # h = 0.5 + A * x, clipped to [0,1]
    return 0.5 + 0.45 * x

def _objective_params(params, beta, lam_sum):
    """
    Objective: minimize LSE of overlaps + sum penalty.
    params = [alpha, phase, A]
    """
    alpha, phase, A = params
    t = np.arange(N, dtype=np.float64)
    frac = np.mod((t + phase) * alpha, 1.0)
    x = 2.0 * frac - 1.0
    h = 0.5 + A * x
    h = np.clip(h, 0, 1)
    
    overlaps = _overlaps_from_h(h)
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _objective_params_grad(params, beta, lam_sum):
    """Same as _objective_params but with gradient hint via finite diff handled by L-BFGS-B."""
    return _objective_params(params, beta, lam_sum)

def construct():
    """
    Returns the optimal sequence of step heights for the Erdos minimum overlap problem.
    
    Strategy: Use quasi-periodic (Sturmian-like) sequences parameterized by slope alpha,
    phase, and amplitude. Optimize these parameters using LSE approximation of max overlap.
    Then perform local refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)  # Should be ~25.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Optimize quasi-periodic parameters (alpha, phase, A)
    # Use multiple betas for annealing effect
    beta_schedule = [5, 10, 20, 40, 80]
    
    # Initial guesses for (alpha, phase, A)
    # Golden ratio conjugate is a natural choice for minimal discrepancy
    golden_conj = (np.sqrt(5) - 1) / 2  # ~0.618
    init_params_list = [
        [golden_conj, 0.0, 0.4],
        [golden_conj, 0.5, 0.4],
        [0.4142, 0.0, 0.4],  # sqrt(2)-1
        [0.4142, 0.5, 0.4],
        [0.3819, 0.0, 0.4],
        [0.3819, 0.5, 0.4],
        [0.5555, 0.0, 0.4],
        [0.5555, 0.5, 0.4],
        [0.4444, 0.0, 0.4],
        [0.4444, 0.5, 0.4],
    ]
    
    bounds_params = [(0.01, 0.99), (0.0, 1.0), (0.05, 0.49)]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        for init_p in init_params_list:
            if time.time() > deadline - 20.0:
                break
            
            try:
                result = minimize(
                    _objective_params,
                    init_p,
                    args=(beta, 500.0),
                    method='L-BFGS-B',
                    bounds=bounds_params,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                alpha, phase, A = result.x
                t = np.arange(N, dtype=np.float64)
                frac = np.mod((t + phase) * alpha, 1.0)
                x = 2.0 * frac - 1.0
                h_new = 0.5 + A * x
                h_new = _project_to_feasible(h_new)
                
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new.copy()
            except Exception:
                pass
    
    # Phase 2: Local refinement using direct h-space optimization with LSE
    if time.time() < deadline - 30.0:
        beta_refine = 50.0
        
        def obj_h(x):
            h = np.clip(x, 0, 1)
            overlaps = _overlaps_from_h(h)
            max_ov = np.max(overlaps)
            if max_ov > 500:
                max_ov = 500
            lse = max_ov + np.log(np.sum(np.exp(beta_refine * (overlaps - max_ov)))) / beta_refine
            sum_penalty = 500.0 * (np.sum(h) - N / 2.0) ** 2
            return lse + sum_penalty
        
        try:
            result = minimize(
                obj_h,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent / swap local search
    if time.time() < deadline - 15.0:
        h_local = best_h.copy()
        local_score = best_score
        improved = True
        max_iters = 2000
        
        it = 0
        while improved and it < max_iters and time.time() < deadline - 15.0:
            improved = False
            it += 1
            # Try adjusting each element slightly
            for i in range(N):
                if time.time() > deadline - 15.0:
                    break
                for delta in [0.01, -0.01, 0.02, -0.02, 0.005, -0.005]:
                    new_h = h_local.copy()
                    new_h[i] = np.clip(new_h[i] + delta, 0, 1)
                    new_h = _project_to_feasible(new_h)
                    new_score = _max_overlap(new_h)
                    if new_score < local_score - 1e-10:
                        h_local = new_h
                        local_score = new_score
                        improved = True
                        break
        
        if local_score < best_score:
            best_score = local_score
            best_h = h_local.copy()
    
    # Final check: ensure sum is exactly N/2
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
