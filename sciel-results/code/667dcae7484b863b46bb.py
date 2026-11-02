"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA_T = (1/4 - (-1/4)) / N  # interval width

def _convolution_fast(h):
    """Compute autoconvolution using FFT for speed.
    
    For piecewise-constant functions on a uniform grid, the continuous autoconvolution
    can be computed via:
    - Pad h to length 2*N-1
    - Use FFT-based convolution
    - The result gives the convolution at grid points, but we need the peak
    
    Actually, for step functions, the autoconvolution is piecewise linear.
    The peak value can be found by evaluating at all relevant points.
    
    A faster approach: use the fact that for step functions, the convolution
    at any point t can be computed as a sum. But the peak is what matters.
    
    Let me use a direct but vectorized approach with a fine grid.
    """
    # Use FFT-based approach for the discrete convolution approximation
    # Then refine to find the actual peak
    
    # The continuous autoconvolution of step functions:
    # (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions with values h[i] on [x_i, x_{i+1}), this is piecewise linear.
    # The maximum occurs at one of the breakpoints or at t=0.
    
    # Key insight: for the autoconvolution of a nonnegative function,
    # the maximum is at t=0 if the function is symmetric, but in general
    # we need to check all critical points.
    
    # Critical points of the autoconvolution are at:
    # t = x_i - x_j for all i, j (i.e., multiples of DELTA_T from -(N-1)*DELTA_T to (N-1)*DELTA_T)
    # and t = 0
    
    # Actually, the breakpoints are at t = k * DELTA_T for k = -(N-1), ..., N-1
    # plus potentially half-integers if the function isn't symmetric.
    
    # For step functions, the autoconvolution is a "tent" shaped function at each 
    # difference. The peak of the total autoconvolution occurs at one of the 
    # grid points t = k * DELTA_T.
    
    # Let's compute the autoconvolution at all grid points t = k * DELTA_T
    # for k = -(N-1), ..., N-1
    
    # At t = k * DELTA_T, the overlap of interval i and interval j shifted by t:
    # Interval i is [x_i, x_{i+1}], interval j shifted by t is [x_j + t, x_{j+1} + t]
    # Overlap length = max(0, min(x_{i+1}, x_{j+1}+t) - max(x_i, x_j+t))
    
    # For t = k * DELTA_T where k is integer:
    # x_i = -1/4 + i * DELTA_T
    # x_{j+1} + t = -1/4 + (j+1) * DELTA_T + k * DELTA_T = -1/4 + (j+1+k) * DELTA_T
    # x_j + t = -1/4 + (j+k) * DELTA_T
    # x_{i+1} = -1/4 + (i+1) * DELTA_T
    # x_i = -1/4 + i * DELTA_T
    # 
    # Overlap = max(0, min((i+1), j+1+k) - max(i, j+k)) * DELTA_T
    #         = DELTA_T * max(0, min(i+1, j+1+k) - max(i, j+k))
    
    # So conv[k*DELTA_T] = DELTA_T * sum_{i,j} h[i]*h[j] * max(0, min(i+1, j+1+k) - max(i, j+k))
    
    # This is equivalent to: for each pair (i,j), the contribution to conv at offset k
    # is h[i]*h[j] * DELTA_T * overlap_count(i, j, k)
    # where overlap_count(i,j,k) = max(0, min(i+1, j+1+k) - max(i, j+k))
    
    # This is the same as the discrete auto-convolution of h with itself,
    # where the "kernel" is a triangle of width 1!
    # 
    # Actually: overlap_count(i,j,k) = 1 - |k - (i-j)| if |k-(i-j)| < 1, else 0
    # Wait, let me recompute.
    # min(i+1, j+1+k) - max(i, j+k)
    # Let d = i - j. Then:
    # min(i+1, j+1+k) = j + min(d+1, 1+k)
    # max(i, j+k) = j + max(d, k)
    # So overlap = min(d+1, 1+k) - max(d, k)
    
    # For integer k and integer d:
    # If d >= k: min(d+1, 1+k) vs max(d,k)=d
    #   If d+1 <= 1+k i.e. d <= k: min = d+1, max = d, overlap = 1
    #   If d+1 > 1+k i.e. d > k: min = 1+k, max = d, overlap = 1+k-d (negative if d>k+1)
    # 
    # Let me just compute it numerically for all k.
    
    n_offsets = 2 * N - 1
    offsets = np.arange(-(N-1), N)  # k values
    
    # For each k, compute conv value
    conv_values = np.zeros(n_offsets)
    
    for idx, k in enumerate(offsets):
        # overlap_count(i,j,k) for all i,j
        # Vectorized computation
        i_vals = np.arange(N)[:, None]  # N x 1
        j_vals = np.arange(N)[None, :]  # 1 x N
        # d = i - j
        # overlap = max(0, min(i+1, j+1+k) - max(i, j+k))
        min_part = np.minimum(i_vals + 1, j_vals + 1 + k)
        max_part = np.maximum(i_vals, j_vals + k)
        overlap = np.maximum(0, min_part - max_part)
        # conv = DELTA_T * sum_{i,j} h[i]*h[j] * overlap[i,j]
        # = DELTA_T * (h[:,None] * h[None,:]).dot(overlap.flatten())
        # More efficiently: DELTA_T * h @ overlap.T @ h  ... no
        # = DELTA_T * sum_i h[i] * sum_j h[j] * overlap[i,j]
        # = DELTA_T * h @ (overlap @ h)  ... let's do:
        # temp = overlap @ h  (N x 1)
        # result = h @ temp
        temp = overlap @ h
        conv_values[idx] = DELTA_T * np.dot(h, temp)
    
    return offsets * DELTA_T, conv_values

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    For a piecewise-constant function with step heights h on N intervals of width DELTA_T:
    - ||f||_2^2 = sum h[i]^2 * DELTA_T
    - ||f||_1 = sum h[i] * DELTA_T  
    - ||f*f||_inf = max of the autoconvolution
    """
    if np.max(h) <= 0:
        return 0.0
    l2_sq = np.sum(h**2) * DELTA_T
    l1 = np.sum(h) * DELTA_T
    t_grid, conv = _convolution_fast(h)
    conv_max = np.max(conv)
    if conv_max <= 0 or l1 <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Key insight: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # This is scale-invariant, so we can normalize ||h||_1 = 1.
    # Then we maximize ||h||_2^2 / ||h*h||_inf.
    # 
    # For a sparse vector with k nonzero entries of equal height 1/k:
    # ||h||_2^2 = k * (1/k)^2 = 1/k
    # ||h*h||_inf depends on the autocorrelation structure.
    # If the nonzero positions form a Sidon set (all pairwise differences distinct),
    # then the autoconvolution peak is minimized.
    #
    # Strategy: search over sparse patterns with positions chosen to minimize
    # the maximum autocorrelation, then optimize the heights.
    
    def score_normalized(h):
        """Score with L1 normalization for stability."""
        if np.max(h) <= 0 or np.sum(h) <= 0:
            return 0.0
        h_norm = h / np.sum(h)
        return _score(h_norm)
    
    # Generate candidate sparse patterns
    def generate_sidon_like_patterns(k):
        """Generate patterns with k nonzero positions that are spread out."""
        patterns = []
        
        # Random Sidon-like sets: positions with distinct differences
        np.random.seed(42)
        for trial in range(200):
            # Start with random positions
            pos = np.random.choice(N, size=k, replace=False)
            # Check if all pairwise differences are distinct
            diffs = set()
            valid = True
            for i in range(k):
                for j in range(i+1, k):
                    d = abs(pos[i] - pos[j])
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                h = np.zeros(N)
                h[pos] = 1.0
                patterns.append(h)
        
        # Also try evenly spaced patterns
        if k <= N:
            step = N // k
            for offset in range(min(step, 5)):
                h = np.zeros(N)
                for i in range(k):
                    pos = offset + i * step
                    if pos < N:
                        h[pos] = 1.0
                if np.sum(h) > 0:
                    patterns.append(h)
        
        # Try prime-based spacing
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in primes:
            if p < N:
                h = np.zeros(N)
                pos = 0
                while pos < N:
                    h[pos] = 1.0
                    pos += p
                if np.sum(h) >= 2:
                    patterns.append(h)
        
        return patterns
    
    # Evaluate various sparsity levels
    best_h = None
    best_score = 0.0
    
    for k in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        if elapsed() > deadline - 5:
            break
        
        patterns = generate_sidon_like_patterns(k)
        
        for h in patterns:
            if elapsed() > deadline - 5:
                break
            s = score_normalized(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # Also try the uniform case
    h_uniform = np.ones(N)
    s_uniform = score_normalized(h_uniform)
    if s_uniform > best_score:
        best_score = s_uniform
        best_h = h_uniform.copy()
    
    # Local optimization from best pattern
    if best_h is not None and elapsed() < deadline - 15:
        def neg_score(h_flat):
            h = np.maximum(h_flat, 0.0)
            if np.sum(h) <= 0:
                return 0.0
            h_norm = h / np.sum(h)
            return -_score(h_norm)
        
        h0 = best_h.copy()
        try:
            res = minimize(
                neg_score,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
            h_opt = np.maximum(res.x, 0.0)
            if np.sum(h_opt) > 0:
                h_opt = h_opt / np.sum(h_opt)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try a few more local optimizations from different good patterns
    if elapsed() < deadline - 30:
        # Get top few patterns
        all_patterns = []
        for k in [2, 3, 4, 5, 6, 7, 8]:
            if elapsed() > deadline - 10:
                break
            patterns = generate_sidon_like_patterns(k)
            all_patterns.extend(patterns)
        
        # Evaluate and sort
        scored_patterns = []
        for h in all_patterns[:50]:
            if elapsed() > deadline - 10:
                break
            s = score_normalized(h)
            scored_patterns.append((s, h))
        
        scored_patterns.sort(key=lambda x: -x[0])
        
        for s_pat, h_pat in scored_patterns[:3]:
            if elapsed() > deadline - 10:
                break
            if s_pat <= best_score:
                continue
            
            def neg_score2(h_flat):
                h = np.maximum(h_flat, 0.0)
                if np.sum(h) <= 0:
                    return 0.0
                h_norm = h / np.sum(h)
                return -_score(h_norm)
            
            try:
                res = minimize(
                    neg_score2,
                    h_pat.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                )
                h_opt = np.maximum(res.x, 0.0)
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt)
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final result
    if best_h is None:
        best_h = np.ones(N)
    
    result = np.maximum(best_h, 0.0).tolist()
    return result
# EVOLVE-BLOCK-END
