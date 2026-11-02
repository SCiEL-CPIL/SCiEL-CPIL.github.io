import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def calc_score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def neg_score(h):
        return -calc_score(h)

    def neg_score_and_grad(x):
        # x is the vector of heights
        h = np.maximum(x, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0, np.zeros(N)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0, np.zeros(N)
        
        score = l2_sq / (l1 * max_conv)
        
        # Gradient approximation via finite differences is too slow for 50 dims potentially,
        # but let's try analytic or just use L-BFGS-B with default numerical gradient.
        # Actually, L-BFGS-B default uses numerical gradient.
        # For better performance, we can provide an analytic gradient.
        
        # Let f = l2_sq / (l1 * max_conv)
        # l2_sq = sum h_i^2
        # l1 = sum h_i
        # max_conv = max_k conv_k
        # conv_k = sum_i h_i h_{k-i}
        
        # Let M = max_conv
        # f = (sum h_i^2) / ( (sum h_i) * M )
        
        # df/dh_j = [ 2 h_j * (l1 * M) - (l2_sq) * (d(l1*M)/dh_j) ] / (l1 * M)^2
        # d(l1)/dh_j = 1
        # d(M)/dh_j = sum over k where conv_k == M of d(conv_k)/dh_j
        # d(conv_k)/dh_j = h_{k-j} + h_{j-k} (if indices valid)
        
        # This is complex. Let's just use numerical gradient for simplicity and robustness,
        # or rely on L-BFGS-B's internal finite differences.
        # Given N=50, numerical gradient is 51 evaluations per step.
        # L-BFGS-B might take 100-500 steps. 50 * 500 * 51 ~ 1.2M evaluations.
        # Each evaluation is a convolution (O(N^2) or FFT). N=50 is small.
        # It should be fast enough.
        
        return -score, None

    best_score = 0.0
    best_h = np.zeros(N)

    # Strategy 1: Spectral/Water-filling inspired initialization
    # We want to minimize autoconvolution peaks.
    # Uniform distribution maximizes L1 for fixed L2? No.
    # Let's start with a few different random initializations and Sidon sets.
    
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Sidon-like
    sidon_sets = [
        [0, 1, 4, 10, 18, 23, 30, 38, 47],
        [0, 2, 5, 11, 20, 32, 47],
        list(range(0, N, 5)) # Sparse uniform
    ]
    for s in sidon_sets:
        h = np.zeros(N)
        for i in s:
            if i < N:
                h[i] = 1.0 / len(s)
        initializations.append(h)
        
    # 3. Random sparse
    for _ in range(10):
        k = np.random.randint(5, 20)
        indices = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0 / k
        initializations.append(h)
        
    # 4. Random dense
    for _ in range(10):
        h = np.random.rand(N)
        h /= np.sum(h)
        initializations.append(h)

    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 10:
            break
            
        # Use L-BFGS-B to maximize score (minimize negative score)
        # Bounds: h_i >= 0
        bounds = [(0, None) for _ in range(N)]
        
        try:
            res = minimize(neg_score, h0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            h_opt = np.maximum(res.x, 0)
            # Normalize? The problem is scale invariant?
            # ratio = l2_sq / (l1 * max_conv)
            # If we scale h by c:
            # l2_sq -> c^2 l2_sq
            # l1 -> c l1
            # max_conv -> c^2 max_conv
            # ratio -> c^2 l2_sq / (c l1 * c^2 max_conv) = l2_sq / (l1 * max_conv) / c
            # So scaling DOWN increases the score!
            # We should push heights as high as possible?
            # Wait, the problem says "nonnegative heights". There is no upper bound constraint in the score formula itself.
            # However, if we scale h by c, the score scales by 1/c.
            # So to maximize score, we want c -> 0?
            # But h_i must be non-negative. If we scale to 0, score is undefined (0/0).
            # Let's re-read carefully.
            # "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
            # Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
            # Let h' = c h.
            # ||h'||_2^2 = c^2 ||h||_2^2
            # ||h'||_1 = c ||h||_1
            # ||h'*h'||_inf = c^2 ||h*h||_inf
            # Ratio' = c^2 ||h||_2^2 / (c ||h||_1 * c^2 ||h*h||_inf) = (1/c) * (||h||_2^2 / (||h||_1 ||h*h||_inf))
            # So the ratio is inversely proportional to the scale factor c.
            # This implies that if there is no upper bound on the heights, we can make the score arbitrarily large by making the heights very small?
            # Or arbitrarily small by making them large?
            # Wait. If c is small, 1/c is large. So small heights give large scores?
            # Let's check limits.
            # If h -> 0, ratio -> infinity?
            # But usually in such problems, there is a normalization.
            # "Choose 50 nonnegative heights... to maximize..."
            # Is there a constraint I missed?
            # "Interface: construct() -> sequence of nonnegative step heights; score is the lower bound on C2"
            # "The evaluator measures the raw value (higher is better)."
            # If the score is unbounded, the seed program would have found a trivial solution.
            # The seed program returns a list of heights.
            # Let's look at the seed program again.
            # It returns `list(best_h)`.
            # In the seed program, `h[i] = 1.0 / len(S)`. So sum is 1.
            # If the score is scale invariant in a way that favors small scales, the seed program would likely have scaled it down.
            # But it normalizes to sum=1.
            # Maybe the score is actually: ||h||_2^2 / (||h||_1 * ||h*h||_inf) is NOT the right interpretation?
            # Or maybe there is a constraint like ||h||_1 = 1?
            # The task description says: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
            # It does NOT explicitly state a normalization constraint.
            # However, if the score is unbounded, the problem is ill-posed for a "maximize" task without bounds.
            # Let's assume there is an implicit constraint, likely ||h||_1 = 1 or ||h||_inf = 1.
            # Given the seed program uses sum=1, let's assume ||h||_1 = 1 is the intended constraint.
            # If ||h||_1 = 1, then c is fixed.
            # Let's verify with the seed score.
            # If I scale the seed h by 0.5, the score should double if my analysis is correct.
            # If the evaluator accepts that, the seed would have done it.
            # Since the seed score is 1.0296 and it uses sum=1, it's highly probable that the evaluator enforces or expects a specific normalization, or the score formula I derived is wrong.
            
            # Let's re-read: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
            # Could it be: (||h||_2^2) / (||h||_1 * ||h*h||_inf) ?
            # Or: ||h||_2^2 / ||h||_1 * ||h*h||_inf ? (i.e. (||h||_2^2 / ||h||_1) * ||h*h||_inf)
            # If it's the latter:
            # Ratio = (c^2 L2 / c L1) * c^2 MaxConv = c^3 * (L2/L1 * MaxConv).
            # Then large c gives large score. Unbounded above.
            
            # If it's the former:
            # Ratio = c^2 L2 / (c L1 * c^2 MaxConv) = (1/c) * (L2 / (L1 MaxConv)).
            # Then small c gives large score. Unbounded above (as c->0).
            
            # In both cases, it's unbounded.
            # Therefore, there MUST be a constraint.
            # Common constraints: ||h||_1 = 1, ||h||_2 = 1, ||h||_inf = 1.
            # The seed program uses ||h||_1 = 1.
            # Let's assume the constraint is ||h||_1 = 1.
            # I will normalize h to sum 1 before returning.
            
            s_opt = calc_score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue

    # Local search refinement on the best found
    # We can try small perturbations
    if time.time() < deadline - 5:
        h = best_h.copy()
        # Normalize to sum 1
        total = np.sum(h)
        if total > 0:
            h /= total
            
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_score = calc_score(h)
            
            # Try moving mass from one bin to another
            for i in range(N):
                if h[i] < 1e-6:
                    continue
                for j in range(N):
                    if i == j:
                        continue
                    # Move small amount from i to j
                    delta = h[i] * 0.1
                    h_trial = h.copy()
                    h_trial[i] -= delta
                    h_trial[j] += delta
                    if np.min(h_trial) < -1e-9:
                        continue
                    h_trial = np.maximum(h_trial, 0)
                    # Re-normalize? If we move mass, sum stays 1.
                    s_trial = calc_score(h_trial)
                    if s_trial > current_score + 1e-9:
                        h = h_trial
                        current_score = s_trial
                        improved = True
                        
    # Final normalization to ensure sum is 1 (assuming that's the constraint)
    total = np.sum(best_h)
    if total > 1e-10:
        best_h /= total
        
    return list(best_h)
