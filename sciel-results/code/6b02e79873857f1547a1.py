"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct step heights h on [0,1] with sum = N/2 (i.e., integral 1 over [0,2] with width 2/N each? 
    Actually, the problem says "step heights h on [0,2], bounded between 0 and 1 with integral 1".
    The interface says: "sequence of step heights in [0, 1] whose sum is exactly len/2".
    
    So we have N steps, each in [0,1], sum = N/2.
    
    We want to minimize the maximum overlap of h with shifted 1-h.
    
    The overlap of h with shifted 1-h: if we define the overlap at shift k as
    sum_i h[i] * (1-h[(i+k) % N]) or similar? 
    
    Actually, "maximum overlap of h with shifted 1-h" likely means:
    For each shift s, compute the sum (or max?) of h[i] * (1-h[i+s]) over i.
    The "maximum overlap" would be the maximum over all shifts s of this quantity.
    
    Wait, let me re-read: "minimize the maximum overlap of h with shifted 1-h".
    
    The overlap between two sequences a and b is typically sum(a[i] * b[i]).
    So for a shift s, the overlap is sum_i h[i] * (1 - h[(i+s) % N]).
    We want to minimize the maximum over s of this overlap.
    
    Alternatively, it could be the max pointwise overlap, but "overlap" usually means inner product.
    
    Let's assume: for each shift s in 0..N-1, compute O(s) = sum_{i=0}^{N-1} h[i] * (1 - h[(i+s) % N]).
    We want to minimize max_s O(s).
    
    Note: O(0) = sum h[i]*(1-h[i]) = sum h[i] - sum h[i]^2 = N/2 - sum h[i]^2.
    
    For h[i] = 0.5 for all i, sum h[i]^2 = N * 0.25 = N/4, so O(0) = N/2 - N/4 = N/4.
    For any other shift s, since h is constant, O(s) = sum 0.5 * 0.5 = N * 0.25 = N/4.
    So max overlap = N/4 = 25 for N=100.
    
    The score is the "upper bound on C5". The evaluator measures the raw value (lower is better).
    The search score is seed/value, so 1.0 equals the seed program.
    
    So if the seed returns max_overlap = 25, then value = 25, and score = seed/25.
    If seed value is also 25, score = 1.0.
    
    We want to find h with sum = N/2, h[i] in [0,1], that minimizes max_s sum_i h[i]*(1-h[(i+s)%N]).
    
    Let's think about what's achievable. If we can make h vary, we might reduce the max overlap.
    
    For example, if h is a "sawtooth" or some other pattern, the overlaps at different shifts might be lower.
    
    Let's use scipy.optimize.minimize to find h that minimizes the max overlap.
    
    Objective: f(h) = max_s sum_i h[i] * (1 - h[(i+s) % N])
    Constraints: sum(h) = N/2, 0 <= h[i] <= 1.
    
    This is a non-convex problem (max of linear functions is convex, but the inner product with 1-h makes it quadratic).
    
    Actually, O(s) = sum h[i] - sum h[i]*h[(i+s)%N] = N/2 - sum h[i]*h[(i+s)%N].
    So minimizing max_s O(s) is equivalent to maximizing min_s sum h[i]*h[(i+s)%N].
    
    So we want to maximize the minimum autocorrelation (with shift).
    
    This is a known problem in design theory: maximize the minimum periodic autocorrelation.
    
    For N=100, we can try to use a numerical optimizer.
    
    Let's set up the optimization:
    Variables: h[0..N-1]
    Objective: -min_s (sum_i h[i]*h[(i+s)%N])  (we minimize the negative of the min autocorrelation)
    Constraints: sum(h) = N/2, 0 <= h[i] <= 1.
    
    We can use SLSQP or trust-constr.
    
    Let's try with a reasonable initial guess and see if it converges within time limits.
    """
    N_local = N
    target_sum = N_local / 2.0
    
    # Initial guess: uniform 0.5
    h0 = np.full(N_local, 0.5)
    
    # Define the objective: minimize -min_s autocorrelation(s)
    # autocorrelation(s) = sum_i h[i] * h[(i+s) % N]
    def objective(h):
        h = np.clip(h, 0, 1)  # ensure bounds during evaluation
        autocorrs = []
        for s in range(N_local):
            shifted = np.roll(h, s)
            ac = np.dot(h, shifted)
            autocorrs.append(ac)
        return -np.min(autocorrs)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - target_sum}
    ]
    
    # Bounds
    bounds = [(0, 1)] * N_local
    
    # Use SLSQP
    result = minimize(
        objective,
        h0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    h_opt = result.x
    # Ensure exact sum and bounds
    h_opt = np.clip(h_opt, 0, 1)
    # Adjust to meet sum exactly
    diff = target_sum - np.sum(h_opt)
    if abs(diff) > 1e-10:
        # Distribute the difference evenly, but respect bounds
        # Simple approach: add diff/N to each, then clip and redistribute
        h_opt = h_opt + diff / N_local
        h_opt = np.clip(h_opt, 0, 1)
        # If still off, adjust the largest/smallest elements
        for _ in range(100):
            current_sum = np.sum(h_opt)
            if abs(current_sum - target_sum) < 1e-12:
                break
            if current_sum < target_sum:
                # Add to elements not at 1
                idx = np.where(h_opt < 1)[0]
                if len(idx) == 0:
                    break
                add = (target_sum - current_sum) / len(idx)
                h_opt[idx] += add
            else:
                # Subtract from elements not at 0
                idx = np.where(h_opt > 0)[0]
                if len(idx) == 0:
                    break
                sub = (current_sum - target_sum) / len(idx)
                h_opt[idx] -= sub
            h_opt = np.clip(h_opt, 0, 1)
    
    return h_opt
# EVOLVE-BLOCK-END
