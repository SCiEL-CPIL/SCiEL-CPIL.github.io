"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    N = len(h)
    overlaps = np.array([np.sum(h * (1 - np.roll(h, k))) for k in range(N)])
    return max(overlaps)

def _score(h):
    """Score is the upper bound on C5, which is the max overlap. Lower is better."""
    return _max_overlap(h)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin

    # Bisection on t: find smallest t such that there exists h in [0,1]^N
    # with sum(h) = N/2 and max_k overlap[k] <= t.
    # The feasibility problem is a QP: minimize 0 subject to:
    #   sum h = N/2
    #   0 <= h <= 1
    #   sum_i h[i]*(1 - h[(i+k)%N]) <= t  for all k
    # which is: sum_i h[i]*h[(i+k)%N] >= N/2 - t  for all k
    # This is a set of quadratic inequality constraints.
    
    # We'll use scipy's minimize to solve the feasibility as a minimization:
    # minimize sum of violations of the overlap constraints
    
    def feasibility_objective(h, t):
        """Return sum of max(0, overlap[k] - t) for all k, plus penalty for constraint violations."""
        h = np.clip(h, 0, 1)
        N_local = len(h)
        overlaps = np.array([np.sum(h * (1 - np.roll(h, k))) for k in range(N_local)])
        violations = np.maximum(0, overlaps - t)
        # Also penalize deviation from sum = N/2
        sum_penalty = (np.sum(h) - N / 2) ** 2
        return np.sum(violations) + 100.0 * sum_penalty

    def feasibility_grad(h, t):
        """Gradient of feasibility objective (approximate via finite differences not needed, use subgradient)."""
        # For simplicity, use SLSQP with analytic gradient via finite diff
        return None

    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try a few structured candidates
    candidates = [best_h.copy()]
    
    # Sinusoidal candidates
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5]:
        h_cand = 0.5 + 0.3 * np.cos(freq * x)
        h_cand = np.clip(h_cand, 0, 1)
        s = np.sum(h_cand)
        if s > 0:
            h_cand = h_cand / s * (N / 2)
            h_cand = np.clip(h_cand, 0, 1)
        s = _score(h_cand)
        candidates.append(h_cand)
        if s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Bisection on t
    t_lo = 0.0
    t_hi = best_score  # We know this is feasible
    
    # First, verify t_hi is feasible
    for _ in range(30):  # Up to 30 bisection steps
        if time.time() > deadline - 10:
            break
        
        t_mid = (t_lo + t_hi) / 2.0
        
        # Try to find feasible h for t_mid using optimization from best_h
        h_init = best_h.copy()
        bounds = [(0, 1)] * N
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2}]
        
        # Use the feasibility objective
        def obj(h):
            return feasibility_objective(h, t_mid)
        
        try:
            res = minimize(obj, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-10})
            
            h_opt = np.clip(res.x, 0, 1)
            # Renormalize
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt / s * (N / 2)
                h_opt = np.clip(h_opt, 0, 1)
            
            # Check if actually feasible
            actual_score = _score(h_opt)
            if actual_score <= t_mid + 1e-6:
                # Feasible, try lower t
                t_hi = t_mid
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_opt.copy()
            else:
                # Not feasible, try higher t
                t_lo = t_mid
        except Exception:
            t_lo = t_mid
    
    # Final validation
    h_final = np.clip(best_h, 0, 1)
    current_sum = np.sum(h_final)
    if current_sum > 0:
        h_final = h_final / current_sum * (N / 2)
        h_final = np.clip(h_final, 0, 1)
    
    return h_final
# EVOLVE-BLOCK-END
