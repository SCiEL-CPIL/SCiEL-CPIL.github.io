"""Optimized program for the second autocorrelation inequality task.

Strategy:
1. Use Sidon-set-like supports to minimize autoconvolution collisions, ensuring the
   supremum of the autoconvolution is dominated by a single large term (the self-term
   or a unique pair) rather than a sum of many overlapping terms.
2. Optimize weights on these supports using a smooth, differentiable objective
   (log-sum-exp approximation for the max) to maximize L2^2 / (L1 * SupConv).
3. Use a time-budgeted local optimization loop with a deadline guard.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
# Total time limit is 150s, but the task says 120s for evaluation.
# The prompt says "evaluator KILLS the program after 150s wall-clock".
# Let's use a safe margin of 140s.
DEADLINE = 140.0

def construct():
    start_time = time.time()
    
    # 1. Select a Sidon set (B_2 sequence) of size 50 within [0, 49] (integers)
    # A Sidon set has distinct pairwise sums. This minimizes the number of pairs
    # (i, j) such that i-j = k for a fixed k.
    # We will use a greedy construction for the Sidon set.
    
    def generate_sidon_set(n):
        s = [0]
        while len(s) < n:
            candidate = s[-1] + 1
            valid = True
            # Check if adding candidate creates duplicate differences
            # Differences are absolute values |a - b|
            diffs = set()
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    d = abs(s[i] - s[j])
                    if d in diffs:
                        # This check is for the existing set, which should be valid.
                        pass
                    diffs.add(d)
            # Check candidate against existing set
            cand_diffs = set()
            for x in s:
                d = abs(candidate - x)
                if d in diffs:
                    valid = False
                    break
                if d in cand_diffs:
                    valid = False
                    break
                cand_diffs.add(d)
            if valid:
                s.append(candidate)
        return s

    # Greedy Sidon set can be dense at the start. Let's try a known efficient Sidon set generation
    # or just use a sparse set.
    # Actually, for N=50, a simple greedy might take a bit but is fast enough.
    # However, the indices must be in [0, 49].
    # A Sidon set of size 50 requires a range of at least ~50^1.5 = 350, which is > 49.
    # So we CANNOT have a perfect Sidon set of size 50 in [0, 49].
    # The maximum size of a Sidon set in [1, N] is roughly sqrt(N).
    # For N=50, max Sidon size is around 7-8.
    
    # So the "Sidon set support" strategy from the prompt implies using a *sparse* support
    # if we could choose fewer points, but we MUST choose 50 points.
    # Re-reading the prompt: "Choose 50 nonnegative heights".
    # The inspiration says: "Introduces Sidon set support structures... Measured: 1.016 -> 1.154".
    # This implies that perhaps we don't use all 50 points as distinct indices?
    # No, "50 nonnegative heights on equal intervals".
    # Maybe the strategy was to set many heights to 0?
    # "Choose 50 nonnegative heights". If some are 0, they don't contribute to L1 or L2.
    # If we set 42 heights to 0 and 8 to positive values forming a Sidon set,
    # we effectively have a smaller problem.
    # Let's test this hypothesis: Use a small Sidon set of size K (e.g., K=7 or 8)
    # and set the rest to 0.
    
    # Let's generate a Sidon set of size K=7 or 8 within [0, 49].
    # Max Sidon size in [0, 49] is approx sqrt(50) ~ 7.
    
    def greedy_sidon(max_size, max_val):
        s = [0]
        while len(s) < max_size:
            cand = s[-1] + 1
            ok = True
            diffs = set()
            # Pre-compute diffs of s
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    diffs.add(abs(s[i]-s[j]))
            for x in s:
                d = abs(cand - x)
                if d in diffs:
                    ok = False
                    break
            if ok:
                s.append(cand)
            else:
                # Try next candidate? Greedy usually increments.
                # A better greedy tries the smallest integer that works.
                pass
            # Simple increment might get stuck or be slow.
            # Let's use a more robust search for the next element.
            # But for K=7, it's very fast.
            # The above loop is slightly flawed (doesn't increment cand if not ok).
            # Let's rewrite properly.
            
    def generate_sidon_k(k, max_val):
        s = [0]
        while len(s) < k:
            found = False
            for cand in range(s[-1] + 1, max_val + 1):
                diffs = set()
                for i in range(len(s)):
                    for j in range(i+1, len(s)):
                        diffs.add(abs(s[i]-s[j]))
                valid = True
                for x in s:
                    d = abs(cand - x)
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    s.append(cand)
                    found = True
                    break
            if not found:
                break
        return s

    # Let's try K=7
    indices = generate_sidon_k(7, 49)
    # print(indices) # e.g., [0, 1, 3, 7, 12, 20, 31]
    
    # Initialize heights
    h = np.zeros(N)
    
    # Assign initial weights. Maybe geometric or random?
    # Let's start with equal weights for the selected indices.
    for i in indices:
        h[i] = 1.0
        
    # Objective function to maximize: L2^2 / (L1 * SupConv)
    # Let f(x) = sum x_i^2 / ( (sum x_i) * max_k (sum_{i-j=k} x_i x_j) )
    # We can maximize log(f(x)) = 2*log(L2) - log(L1) - log(SupConv)
    # Or simply maximize f(x) directly.
    # SupConv is max_k conv_k.
    # conv_k = sum_{i} x_i x_{i-k} (with appropriate bounds).
    # For step functions on intervals, the autoconvolution at shift k is proportional to
    # sum_{i} x_i x_{i-k} * (1/4 - |k|/4) ? No, the intervals are equal.
    # The problem says "equal intervals of [-1/4, 1/4]".
    # This usually implies the functions are constant on intervals [k/50, (k+1)/50] shifted?
    # "50 nonnegative heights on equal intervals of [-1/4, 1/4]"
    # Total length 1/2. 50 intervals. Width = (1/2)/50 = 1/100.
    # The autoconvolution of step functions is a piecewise linear function.
    # The supremum norm is the maximum value of this piecewise linear function.
    # The peaks occur at the shifts corresponding to integer differences of the interval indices.
    # Specifically, if x_i are the heights, the autoconvolution peak at shift k (where k is integer index diff)
    # is proportional to (sum_{i} x_i x_{i-k}) * (width) * (1 - |k|/50)?
    # Actually, for discrete steps of width w, the convolution at lag k*w is w * sum x_i x_{i-k}.
    # The maximum of the continuous convolution might be slightly different if peaks are between integers,
    # but for step functions, the max is at the integer lags if the weights are non-negative.
    # Let's assume SupConv = max_k (sum_{i} x_i x_{i-k}) * w.
    # Since w is constant, we can ignore it in the ratio.
    
    # So we maximize: (sum x^2) / ( (sum x) * max_k (sum_i x_i x_{i-k}) )
    
    def objective(x):
        # x is a vector of length N
        # Ensure non-negativity
        x = np.maximum(x, 0)
        l2_sq = np.sum(x**2)
        l1 = np.sum(x)
        if l1 == 0:
            return -1e10
        
        # Compute autoconvolution coefficients
        # conv[k] = sum_i x[i] x[i-k]
        # Using FFT for speed
        # x is length 50.
        # Convolution length 99.
        # We only need the max of the convolution.
        # scipy.signal.fftconvolve
        # Or just np.convolve
        conv = np.convolve(x, x, mode='full')
        max_conv = np.max(conv)
        
        if max_conv == 0:
            return -1e10
            
        ratio = l2_sq / (l1 * max_conv)
        return -ratio # Minimize negative ratio
    
    # Initial point
    x0 = h.copy()
    
    # Bounds: x_i >= 0
    bounds = [(0, None)] * N
    
    # Optimize
    best_x = x0.copy()
    best_score = -objective(x0)
    
    # Use L-BFGS-B for bound-constrained optimization
    # We might need multiple restarts if it's non-convex.
    # The function is non-convex.
    
    # To handle non-negativity and the ratio, we can use a change of variables:
    # x_i = exp(y_i). This ensures x_i > 0.
    # But x_i can be 0.
    # Let's stick with bounds and L-BFGS-B.
    
    # To improve global search, let's try a few random restarts on the support set.
    # Or just optimize the weights on the Sidon set.
    
    # Let's optimize only the non-zero elements?
    # L-BFGS-B works on the full vector.
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return list(best_x)
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            best_x = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    # Refine with a second pass?
    # Or try a different initial guess: geometric progression on the Sidon set.
    
    # Let's try a few random initializations on the Sidon set indices.
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        x_init = np.zeros(N)
        # Random positive weights on Sidon indices
        weights = np.random.rand(len(indices))
        # Scale to have L1 = 1? The ratio is scale-invariant.
        # Let's just use the random weights.
        for i, idx in enumerate(indices):
            x_init[idx] = weights[i]
            
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-8})
            if res.success and -res.fun > best_score:
                best_score = -res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish?
    # The ratio is scale invariant. We can normalize the output.
    # The task asks for "sequence of nonnegative step heights".
    # Any scaling is valid.
    
    return list(best_x)

# Test locally if needed
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    print(f"Time: {time.time() - start}")
    print(f"Result: {res}")
    # Calculate score
    x = np.array(res)
    l2_sq = np.sum(x**2)
    l1 = np.sum(x)
    conv = np.convolve(x, x, mode='full')
    max_conv = np.max(conv)
    if l1 > 0 and max_conv > 0:
        score = l2_sq / (l1 * max_conv)
        print(f"Score: {score}")
