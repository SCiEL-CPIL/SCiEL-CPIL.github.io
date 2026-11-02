import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 135.0

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    if n == 0:
        return np.inf
    
    # Linear autoconvolution via FFT
    M = 1 << (2*n - 1).bit_length()
    h_padded = np.zeros(M)
    h_padded[:n] = heights
    H = np.fft.fft(h_padded)
    auto_conv = np.fft.ifft(H * H).real[:2*n-1]
    
    total = np.sum(heights)
    if total <= 0:
        return np.inf
    
    # Normalization: for uniform heights=1, auto_conv[0]=1, total=N
    # score = max(auto_conv) * n / total^2 gives 1.0 for uniform
    score = np.max(auto_conv) * n / (total * total)
    return score

def _smooth_max_and_grad(auto_conv, beta=50.0):
    """Compute smooth max of autoconvolution and gradient weights."""
    max_ac = np.max(auto_conv)
    shifted = beta * (auto_conv - max_ac)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    smooth_max = max_ac + np.log(sum_exp) / beta
    return smooth_max, weights

def _compute_gradient(h, weights, n):
    """Compute gradient of smooth max w.r.t. h using FFT-based approach.
    
    dc[k]/dh[i] = (h[k-i] + h[i]) if 0 <= k-i < n else 0
    
    grad[i] = sum_k weights[k] * (h[k-i] + h[i]) * 1_{0<=k-i<n}
    = sum_k weights[k] * h[k-i] * 1_{0<=k-i<n} + h[i] * sum_k weights[k] * 1_{0<=k-i<n}
    """
    # First term: sum_k weights[k] * h[k-i] * 1_{0<=k-i<n}
    # Let j = k-i, so k = j+i. Sum over j from 0 to n-1 where j+i < 2n-1
    # This is a cross-correlation: for each i, sum_{j=0}^{n-1} h[j] * weights[j+i]
    
    # Build extended weights array
    w_ext = np.zeros(2*n-1)
    w_ext[:] = weights
    
    # Use FFT-based approach
    M2 = 1 << (2*n).bit_length()
    h_pad = np.zeros(M2)
    h_pad[:n] = h
    
    # For the first term, we need for each i: sum_j h[j] * w_ext[i+j]
    # This is a convolution-like operation
    
    # Actually, let's use a different approach:
    # grad_first[i] = sum_{j=0}^{n-1} h[j] * weights[j+i] (where j+i < 2n-1)
    
    # We can compute this as a cross-correlation using FFT
    # Create a reversed h and convolve with weights
    h_rev = h[::-1]
    
    # Cross-correlation: (h * weights)[i] = sum_j h[j] * weights[i-j]
    # But we want sum_j h[j] * weights[j+i]
    # Let's flip weights: w_flip[k] = weights[2n-2-k]
    # Then sum_j h[j] * weights[j+i] = sum_j h[j] * w_flip[2n-2-j-i]
    # This is getting complicated. Let's just do it directly for n=600.
    
    # For n=600, the direct computation is 600*600 = 360K ops, which is fast.
    grad_first = np.zeros(n)
    for i in range(n):
        # j from 0 to min(n-1, 2n-2-i)
        max_j = min(n, 2*n-1-i)
        if max_j > 0:
            grad_first[i] = np.dot(h[:max_j], w_ext[i:i+max_j])
    
    # Second term: h[i] * sum_{j=0}^{n-1} weights[j+i] * 1_{j+i < 2n-1}
    grad_second = np.zeros(n)
    for i in range(n):
        max_j = min(n, 2*n-1-i)
        if max_j > 0:
            grad_second[i] = h[i] * np.sum(w_ext[i:i+max_j])
    
    return grad_first + grad_second

def _objective_and_grad(params, beta=50.0):
    """Objective: smooth max of linear autoconvolution, normalized.
    Returns (score, gradient).
    """
    h = np.asarray(params, dtype=np.float64)
    h = np.maximum(h, 0.0)
    n = len(h)
    
    # Compute linear autoconvolution via FFT
    M = 1 << (2*n - 1).bit_length()
    h_padded = np.zeros(M)
    h_padded[:n] = h
    H = np.fft.fft(h_padded)
    auto_conv = np.fft.ifft(H * H).real[:2*n-1]
    
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    smooth_max, weights = _smooth_max_and_grad(auto_conv, beta)
    score = smooth_max * n / (total * total)
    
    # Gradient
    d_smooth_max_dh = _compute_gradient(h, weights, n)
    
    # d(score)/dh[i] = n/(total^2) * d(smooth_max)/dh[i] - 2*n*smooth_max/(total^3)
    d_score_dh = (n / (total*total)) * d_smooth_max_dh - (2*n*smooth_max / (total**3))
    
    # Project to non-negative: zero out gradient where h is at boundary
    d_score_dh[h <= 0] = 0
    
    return score, d_score_dh

def _two_block_optimize():
    """Optimize a two-block step function: h = [a]*k + [b]*(n-k).
    Returns best (score, heights).
    """
    best_score = np.inf
    best_h = np.ones(N)
    
    # Try different split points and ratios
    for k in range(N//4, 3*N//4):
        for ratio in np.linspace(0.1, 10.0, 50):
            h = np.zeros(N)
            h[:k] = ratio
            h[k:] = 1.0
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Also try the reverse
    for k in range(N//4, 3*N//4):
        for ratio in np.linspace(0.1, 10.0, 50):
            h = np.zeros(N)
            h[:k] = 1.0
            h[k:] = ratio
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_score, best_h

def _bisection_lp_optimize():
    """Use bisection on t with LP feasibility check.
    
    Minimize t subject to:
    - auto_conv[k] <= t * total^2 / n for all k
    - h >= 0
    - sum(h) = 1 (normalize)
    
    This is a quadratic constraint, so LP won't work directly.
    Instead, we'll use a different approach: fix total=1 and minimize max(auto_conv).
    """
    # For now, just return the uniform solution as a fallback
    h = np.ones(N)
    return compute_score(h), h

def _coordinate_descent(h, max_iter=1000, tol=1e-10, time_limit=30.0):
    """Improved coordinate descent with geometric line search including zero."""
    start_time = time.time()
    h = h.copy()
    n = len(h)
    current_score = compute_score(h)
    
    # Precompute the autoconvolution
    def get_autoconv(hh):
        M = 1 << (2*n - 1).bit_length()
        h_pad = np.zeros(M)
        h_pad[:n] = hh
        H = np.fft.fft(h_pad)
        return np.fft.ifft(H * H).real[:2*n-1]
    
    # Geometrically spaced candidates around current value, including zero
    def candidate_values(val):
        if val <= 1e-15:
            return [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
        else:
            return [0.0] + [val * r for r in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]]
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                    break
            
            old_val = h[i]
            best_val = old_val
            best_score = current_score
            
            for cand in candidate_values(old_val):
                h[i] = cand
                new_score = compute_score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_val = cand
            
            if best_val != old_val:
                h[i] = best_val
                current_score = best_score
                improved = True
        
        if not improved:
            break
    
    return h, current_score

def _zero_out_pass(h, threshold=0.001):
    """Remove small values to reduce noise and potentially improve score."""
    h = h.copy()
    n = len(h)
    current_score = compute_score(h)
    
    # Try zeroing out small values
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05]:
        h_trial = h.copy()
        h_trial[h_trial < thresh] = 0.0
        
        if np.sum(h_trial) > 0:
            score_trial = compute_score(h_trial)
            if score_trial < current_score:
                h = h_trial
                current_score = score_trial
    
    return h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start from uniform
    h0 = np.ones(N)
    best_h = h0.copy()
    best_score = compute_score(h0)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Two-block optimization for a good starting point
    if time_left() > 100:
        two_block_score, two_block_h = _two_block_optimize()
        if two_block_score < best_score:
            best_score = two_block_score
            best_h = two_block_h.copy()
    
    # Phase 2: Full optimization from multiple starting points
    n_starts = 5
    for start_idx in range(n_starts):
        if time_left() < 30:
            break
        
        if start_idx == 0:
            h_init = best_h.copy()
        elif start_idx == 1:
            h_init = h0.copy()
        else:
            # Perturbed start
            h_init = best_h.copy()
            freqs = np.arange(N)
            for _ in range(3):
                k = rng.integers(1, N//4)
                phase = rng.uniform(0, 2*np.pi)
                amp = rng.uniform(0.05, 0.15)
                h_init += amp * np.cos(2*np.pi*k*freqs/N + phase)
            h_init = np.maximum(h_init, 0.0)
        
        # Optimize
        try:
            result = minimize(
                _objective_and_grad,
                h_init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            h_opt = np.maximum(result.x, 0.0)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent with geometric line search
    if time_left() > 40:
        cd_h, cd_score = _coordinate_descent(best_h, max_iter=500, time_limit=25.0)
        if cd_score < best_score:
            best_score = cd_score
            best_h = cd_h.copy()
    
    # Phase 4: Zero-out pass to remove noise
    if time_left() > 25:
        zo_h, zo_score = _zero_out_pass(best_h)
        if zo_score < best_score:
            best_score = zo_score
            best_h = zo_h.copy()
    
    # Phase 5: Local refinement with smaller steps
    if time_left() > 20:
        for _ in range(30):
            if time_left() < 15:
                break
            h_trial = best_h + rng.normal(0, 0.005, size=N)
            h_trial = np.maximum(h_trial, 0.0)
            score_trial = compute_score(h_trial)
            if score_trial < best_score:
                best_score = score_trial
                best_h = h_trial.copy()
    
    # Phase 6: Final polish with tighter tolerance and higher beta
    if time_left() > 15:
        try:
            # Use higher beta for sharper max approximation
            result = minimize(
                lambda x: _objective_and_grad(x, beta=300.0)[0],
                best_h.copy(),
                jac=lambda x: _objective_and_grad(x, beta=300.0)[1],
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            h_opt = np.maximum(result.x, 0.0)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final zero-out and coordinate descent polish
    if time_left() > 10:
        final_h, final_score = _zero_out_pass(best_h)
        if final_score < best_score:
            best_score = final_score
            best_h = final_h.copy()
        
        if time_left() > 8:
            cd_h, cd_score = _coordinate_descent(best_h, max_iter=200, time_limit=6.0)
            if cd_score < best_score:
                best_score = cd_score
                best_h = cd_h.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
