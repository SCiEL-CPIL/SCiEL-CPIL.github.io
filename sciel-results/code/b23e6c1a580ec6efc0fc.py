"""
Strategy: Combine Sidon set support with continuous weight optimization.

The current program uses parametric shapes. The inspirations suggest that
sparse supports (Sidon sets) are superior because they minimize the
autoconvolution peak (off-diagonal terms).

We combine these:
1. Use the strong Sidon set support idea from Inspiration 1 & 2.
2. Instead of fixed uniform weights (1/k), use a continuous optimization
   (Scipy Minimize) to find the optimal weights on that support.
3. Use a smooth surrogate (Log-Sum-Exp) for the max operation in the
   denominator to allow gradient-based optimization.
4. Perform this optimization for several candidate Sidon sets (various
   sizes and random restarts) and keep the best result.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Note: The problem statement implies the full convolution max, but
    typically for these inequalities, the 'peak' refers to the off-diagonal
    or the global max. The seed program uses full max.
    However, Inspiration 2 excludes lag 0. Let's stick to the standard
    definition of autoconvolution supremum norm which includes all lags.
    If we exclude lag 0, the ratio can be arbitrarily large by making
    the signal sparse? No, if h is a delta function, conv is a delta.
    Max is 1. L1 is 1. L2^2 is 1. Ratio 1.
    If we exclude lag 0, and h is a delta, max_offpeak is 0. Ratio inf.
    The task says "supremum norm of the autoconvolution". Usually this is
    the max over all lags.
    The seed program uses `np.max(hh)` where hh is full convolution.
    So we must include lag 0.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = fftconvolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set(n, target_size, rng):
    """Find a Sidon set of size target_size in [0, n) using random greedy."""
    best = []
    for _ in range(200): # Random restarts
        current = []
        sums = set()
        # Random order of indices
        indices = list(range(n))
        rng.shuffle(indices)
        for i in indices:
            can_add = True
            # New sums: i+i, i+j for j in current
            new_sums = [2*i]
            for j in current:
                new_sums.append(i+j)
            
            # Check for collisions
            for s in new_sums:
                if s in sums:
                    can_add = False
                    break
            
            if can_add:
                current.append(i)
                for s in new_sums:
                    sums.add(s)
                
                if len(current) >= target_size:
                    break
        
        if len(current) > len(best):
            best = current
        if len(best) >= target_size:
            break
    return best

def optimize_weights_for_support(support, beta=50.0, max_iter=500):
    """
    Optimize weights h_i for i in support to maximize ratio.
    Objective: Maximize log(L2^2) - log(L1 * MaxConv)
    Since L1 is fixed (we normalize weights to sum to 1), we maximize log(L2^2) - log(MaxConv).
    Or minimize -log(L2^2) + log(MaxConv).
    
    We use LogSumExp for MaxConv to make it smooth.
    MaxConv = max_k conv_k.
    log(MaxConv) approx (1/beta) * log(sum(exp(beta * conv_k))).
    
    Variables: x (log-weights) to ensure positivity.
    Weights w = softmax(x).
    h is zero except at support indices, where it is w.
    We normalize h so that sum(h) = 1. Since w sums to 1, h already sums to 1.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights
    x0 = np.zeros(k)
    
    def objective_and_grad(x):
        w = np.exp(x - np.max(x))
        w /= np.sum(w)
        
        h = np.zeros(N)
        h[support] = w
        
        # Compute convolution
        conv = fftconvolve(h, h, mode='full')
        
        # L2 norm squared
        l2_sq = np.sum(h * h)
        if l2_sq < 1e-15:
            l2_sq = 1e-15
        log_l2_sq = np.log(l2_sq)
        
        # LogSumExp for max conv
        # We want to maximize L2^2 / (L1 * MaxConv).
        # Minimize -log(L2^2) + log(MaxConv).
        # log(MaxConv) approx (1/beta) log(sum(exp(beta conv)))
        
        # To avoid overflow in exp(beta * conv), shift by max
        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        log_sum_exp = max_conv + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
        
        val = -log_l2_sq + log_sum_exp
        
        # Gradient
        # d/dx (-log_l2_sq) + d/dx (log_sum_exp)
        
        # 1. Gradient of log_sum_exp w.r.t x
        # log_sum_exp = max_conv + (1/beta) log(sum(exp(beta (conv - max_conv))))
        # Let S = sum(exp(beta (conv - max_conv)))
        # d log_sum_exp / d conv_k = (1/S) exp(beta (conv_k - max_conv)) = softmax(beta (conv - max_conv))
        # Let p_k be these softmax weights.
        # d log_sum_exp / d h_j = sum_k p_k * (dh_k/dh_j)
        # conv_k = sum_i h_i h_{k-i}
        # dh_k/dh_j = h_{k-j} + h_{j-k} (with boundary checks, 0 if out of bounds)
        
        # 2. Gradient of -log_l2_sq w.r.t x
        # l2_sq = sum h_i^2 = sum w_i^2 (since h is non-zero only at support)
        # d log_l2_sq / d w_j = 2 w_j / l2_sq
        # d w_j / d x_m = w_j (delta_jm - w_m)
        # So d log_l2_sq / d x_m = sum_j (2 w_j / l2_sq) w_j (delta_jm - w_m)
        # = (2 w_m / l2_sq) (1 - sum_j w_j^2) = (2 w_m / l2_sq) (1 - l2_sq)
        
        # Let's compute gradients w.r.t w first, then chain rule.
        grad_w = np.zeros(k)
        
        # Part 1: Convolution gradient
        # p_k = softmax(beta * (conv - max_conv))
        exp_shifted = np.exp(shifted)
        S = np.sum(exp_shifted)
        p = exp_shifted / S
        
        # grad_conv w.r.t h
        # We need grad_w[m] = sum_j (d log_sum_exp / d h_j) * (d h_j / d w_m)
        # h_j is 0 if j not in support, w_m if j == support[m].
        # So d h_j / d w_m is 1 if j == support[m], 0 otherwise.
        # Thus grad_w[m] = d log_sum_exp / d h_{support[m]}
        
        # d log_sum_exp / d h_j = sum_k p_k * (dh_k/dh_j)
        # dh_k/dh_j = h_{k-j} + h_{j-k}
        
        # Efficient calculation:
        # For each m, we need sum_k p_k * (h_{k - support[m]} + h_{support[m] - k})
        # This is essentially (p * h)_{support[m]} + (p_rev * h)_{support[m]}?
        # Actually, conv is linear in h.
        # Let's just loop over m (k is up to ~12).
        
        for m in range(k):
            j = support[m]
            # Sum over all lags k in conv
            # h_{k-j} is non-zero only if k-j is in support.
            # h_{j-k} is non-zero only if j-k is in support.
            
            # Term 1: sum_k p_k h_{k-j}
            # Let l = k-j => k = l+j.
            # Sum_l p_{l+j} h_l
            # h_l is non-zero only if l is in support.
            
            # Term 2: sum_k p_k h_{j-k}
            # Let l = j-k => k = j-l.
            # Sum_l p_{j-l} h_l
            
            # We can precompute the contribution of each h_l to the gradient of log_sum_exp.
            # G_l = d log_sum_exp / d h_l = sum_k p_k (delta_{k, l+j} + delta_{k, j-l})? No.
            # G_l = sum_k p_k (dh_k/dh_l)
            # dh_k/dh_l = h_{k-l} + h_{l-k}
            # This is getting complicated to vectorize simply without full matrix.
            # Given k is small, we can just do O(k * N_conv) which is small.
            
            grad_val = 0.0
            # Iterate over all lags in conv
            # conv index r corresponds to lag r - (N-1)
            # h is defined on 0..N-1.
            # conv_r = sum_i h_i h_{r-i}
            # d conv_r / d h_j = h_{r-j} + h_{j-r}
            
            for r in range(len(conv)):
                if p[r] == 0: continue
                # h_{r-j}
                idx1 = r - j
                if 0 <= idx1 < N:
                    h_val1 = h[idx1]
                else:
                    h_val1 = 0.0
                # h_{j-r}
                idx2 = j - r
                if 0 <= idx2 < N:
                    h_val2 = h[idx2]
                else:
                    h_val2 = 0.0
                grad_val += p[r] * (h_val1 + h_val2)
            
            grad_w[m] = grad_val
        
        # Part 2: L2 term
        # d(-log_l2_sq)/d w_m = - (2 w_m / l2_sq) (1 - l2_sq)
        factor = -(2.0 / l2_sq) * (1.0 - l2_sq)
        grad_w += factor * w
        
        # Chain rule: d/dx = d/dw * dw/dx
        # dw_m/dx_n = w_m (delta_mn - w_n)
        # grad_x_n = sum_m grad_w_m w_m (delta_mn - w_n)
        # = w_n (sum_m grad_w_m w_m - w_n sum_m grad_w_m w_m) ? No.
        # grad_x = w * (grad_w - dot(grad_w, w))
        
        dot_gw_w = np.dot(grad_w, w)
        grad_x = w * (grad_w - dot_gw_w)
        
        return val, grad_x
    
    # Optimize
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': max_iter, 'ftol': 1e-10})
    
    w_opt = np.exp(res.x - np.max(res.x))
    w_opt /= np.sum(w_opt)
    
    h_opt = np.zeros(N)
    h_opt[support] = w_opt
    
    score = compute_ratio(h_opt)
    return h_opt, score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Generate candidate Sidon sets
    candidates = []
    
    # 1. Try to find large Sidon sets
    # Max size is roughly sqrt(N) ~ 7. Let's try up to 12 to be safe, but likely 8-10 is max.
    for k in range(12, 3, -1):
        if time.time() > deadline:
            break
        found = find_sidon_set(N, k, rng)
        if len(found) == k:
            candidates.append(found)
            # Add a few random variations of this size
            for _ in range(5):
                if time.time() > deadline:
                    break
                f2 = find_sidon_set(N, k, rng)
                if len(f2) == k and f2 not in candidates:
                    candidates.append(f2)
    
    # 2. Add some random small sets
    for k in range(4, 8):
        for _ in range(10):
            if time.time() > deadline:
                break
            indices = sorted(rng.choice(N, k, replace=False).tolist())
            if is_sidon_set(indices):
                candidates.append(indices)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Sort by size descending? Larger sets usually better for L2/L1, but might increase conv peak.
    # Let's just try them all.
    
    for support in unique_candidates:
        if time.time() > deadline:
            break
        
        h_opt, score = optimize_weights_for_support(support)
        
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Final fallback if no improvement
    if best_score <= 1.0:
        # Try a simple uniform
        h_uniform = np.ones(N)
        s = compute_ratio(h_uniform)
        if s > best_score:
            best_h = h_uniform
            best_score = s
            
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
