import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _autoconv(heights):
    """Compute autoconvolution using FFT."""
    n = len(heights)
    M = 1
    while M < 2 * n - 1:
        M *= 2
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    return conv

def _log_domain_bfgs(h_init, max_iters=100, alpha=100.0, mu=100.0):
    """
    BFGS optimization in log-domain.
    Objective: Minimize smooth_max(conv(exp(x))) + mu * (sum(exp(x)) - 1)^2
    """
    n = len(h_init)
    h = np.maximum(h_init, 1e-10)
    # Normalize initial h to sum to 1
    h = h / np.sum(h)
    
    x = np.log(h)
    
    # BFGS state
    B = np.eye(n)
    s_prev = None
    y_prev = None
    
    # Initial gradient
    def objective_and_grad(x):
        h = np.exp(x)
        # Penalty for sum constraint
        sum_h = np.sum(h)
        penalty = mu * (sum_h - 1.0) ** 2
        pen_grad = 2 * mu * (sum_h - 1.0) * h
        
        # Smooth max of conv
        conv = _autoconv(h)
        max_conv = np.max(conv)
        # Log-sum-exp for smooth max
        # phi = max_conv + (1/alpha) * log(sum(exp(alpha * (conv - max_conv))))
        shifted = alpha * (conv - max_conv)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        smooth_max = max_conv + log_sum / alpha
        
        # Gradient of smooth max
        # d(smooth_max)/dh_i = (1/sum_exp) * sum_k exp(alpha*conv_k) * d(conv_k)/dh_i
        # Weights for each k in the max
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        
        # d(conv_k)/dh_i = h_{k-i} + h_{i-k}
        # We need sum_k weights_k * (h_{k-i} + h_{i-k})
        # This is equivalent to convolving weights with h, but weights are length 2n-1
        # Let's compute gradient vector g_h of length n
        g_h = np.zeros(n)
        
        # Efficient computation of gradient of smooth max
        # grad_h[i] = sum_k w_k * (h[k-i] + h[i-k])
        # Note: h is defined on [0, n). conv is defined on [0, 2n-2].
        # For a given i, h[k-i] is non-zero if 0 <= k-i < n => i <= k < i+n.
        # h[i-k] is non-zero if 0 <= i-k < n => i-n < k <= i.
        
        # We can compute this by convolution of weights (padded) with h?
        # Actually, let's just do it via FFT or direct if n is small enough?
        # N=600 is small enough for O(N^2) if we are careful, but FFT is better.
        
        # Let's use the property:
        # grad_h = conv(w, h) + conv(w_rev, h) ... no, indices are tricky.
        
        # Simpler: 
        # grad_h[i] = sum_{k} w_k h_{k-i} + sum_{k} w_k h_{i-k}
        # Let u = k-i => k = u+i. Sum over u such that 0<=u<n and 0<=u+i<2n-1.
        # This is essentially a cross-correlation.
        
        # Let's just compute it directly for clarity and correctness, N=600 is fast enough for 100 iters?
        # 600*600 = 360k ops per gradient. 100 iters = 36M ops. Python loop is too slow.
        # Use FFT.
        
        # Term 1: sum_k w_k h_{k-i}
        # Let W be FFT of w (padded to 2n). H be FFT of h (padded to 2n).
        # We want correlation.
        
        # Actually, we can compute the gradient of the quadratic form h^T A h where A depends on weights?
        # No, weights depend on h.
        
        # Let's use a numerical gradient for the smooth max part if analytical is too complex?
        # No, we need speed.
        
        # Analytical gradient of smooth max:
        # Let S = sum_k exp(alpha * conv_k).
        # d(smooth_max)/dh_i = (1/S) * sum_k exp(alpha * conv_k) * (h_{k-i} + h_{i-k})
        
        # We can compute the vector V_i = sum_k w_k (h_{k-i} + h_{i-k})
        # Note that w_k is the weight for conv_k.
        # conv_k = sum_j h_j h_{k-j}.
        
        # Let's compute the gradient using FFT of the weights.
        # We need to correlate w with h.
        
        M = 1
        while M < 2 * n - 1:
            M *= 2
            
        # Pad w to length M
        w_padded = np.zeros(M)
        w_padded[:2*n-1] = weights
        
        W = np.fft.rfft(w_padded)
        H = np.fft.rfft(h)
        
        # Correlation of w and h
        # corr[i] = sum_k w_k h_{k-i}
        # This is equivalent to conv(w_rev, h) ?
        # Let's check: (w_rev * h)[i] = sum_k w_rev[k] h[i-k] = sum_k w[M-1-k] h[i-k].
        # If we define w_rev properly...
        
        # Simpler:
        # grad_h[i] = sum_k w_k h_{k-i} + sum_k w_k h_{i-k}
        
        # Let's compute term1[i] = sum_k w_k h_{k-i}
        # This is the cross-correlation of w and h.
        # In FFT domain: IFFT(FFT(w) * conj(FFT(h))) ? No, that's for complex.
        # For real signals, correlation can be done via IFFT(FFT(w) * conj(FFT(h))) if we handle padding correctly.
        
        # Let's just use the direct sum for the gradient of the smooth max part?
        # N=600. 100 iterations.
        # 600 * 600 * 100 = 36,000,000 operations.
        # In Python, a loop of 36M ops takes ~1-2 seconds.
        # This is acceptable within 140s budget if we have few restarts.
        
        # However, we can do better.
        # Let's compute the gradient via FFT.
        
        # term1 = ifft(fft(w) * conj(fft(h)))
        # term2 = ifft(fft(w) * fft(h)) ? No.
        
        # Let's stick to the analytical formula:
        # grad_h[i] = sum_k w_k (h_{k-i} + h_{i-k})
        
        # We can compute this by convolving w with h and w with h_rev?
        
        # Let's try a hybrid: compute weights, then use FFT to compute the weighted sum.
        
        # Actually, there is a simpler way.
        # The gradient of the max of a set of quadratic forms is a weighted sum of their gradients.
        # grad conv_k = 2 * (shifted_h + reversed_shifted_h)
        
        # Let's compute the gradient vector g_h
        g_h = np.zeros(n)
        
        # Weights are on indices 0..2n-2
        # For each i in 0..n-1:
        # g_h[i] = sum_{k} w_k (h_{k-i} + h_{i-k})
        
        # Let's compute this using FFT.
        # Let W be the FFT of w (padded to M).
        # Let H be the FFT of h (padded to M).
        
        # The term sum_k w_k h_{k-i} is the cross-correlation.
        # Cross-correlation c[i] = sum_k w_k h_{k-i} = (w * h_rev)[i]?
        # Let's just use numpy correlate? No, too slow.
        
        # Let's use the FFT property:
        # IFFT(FFT(w) * conj(FFT(h))) gives the cross-correlation?
        # For real signals, yes, if padded correctly.
        
        W = np.fft.rfft(w_padded)
        H = np.fft.rfft(h)
        
        # Correlation w, h
        corr_wh = np.fft.irfft(W * np.conj(H), n=M)
        # corr_wh[i] = sum_k w_k h_{k-i} ?
        # Let's verify with small example.
        
        # The other term sum_k w_k h_{i-k} is sum_k w_k h_{-(k-i)} = sum_k w_k h_{i-k}.
        # This is the correlation of w and h, but with h reversed?
        # Actually, h_{i-k} = h_rev[k-i] if h_rev[j] = h[n-1-j]? No.
        
        # Let's just compute the two terms separately using FFT.
        # Term 1: sum_k w_k h_{k-i}
        # This is the standard cross-correlation.
        # Term 2: sum_k w_k h_{i-k}
        # This is sum_k w_k h_{-(k-i)}.
        
        # Note that h is zero outside [0, n).
        
        # Let's compute Term 1:
        t1 = np.fft.irfft(W * np.conj(H), n=M)
        # t1[i] should be sum_k w_k h_{k-i}
        
        # Let's compute Term 2:
        # h_{i-k} = h[-(k-i)].
        # Let h_neg[j] = h[-j] (with zero padding).
        # Then Term 2 is sum_k w_k h_neg[k-i].
        # This is the cross-correlation of w and h_neg.
        
        h_neg = np.zeros(M)
        h_neg[:n] = h[::-1] # h_neg[j] = h[n-1-j]? No, we want h[-j].
        # h is index 0..n-1.
        # h[-j] is h[n-j] if we wrap? No, h is 0 outside.
        # h[-j] is non-zero only if -j is in [0, n-1] => j in [-n+1, 0].
        # So h_neg[j] = h[-j] for j in [-n+1, 0].
        # In an array of size M, indices 0..M-1.
        # We want h_neg[k-i] to be h[i-k].
        # If i-k is in [0, n-1], then h[i-k] is the value.
        # So h_neg[j] = h[j] if j >= 0? No.
        
        # Let's just use the fact that h is real and symmetric? No.
        
        # Alternative:
        # grad_h[i] = sum_k w_k h_{k-i} + sum_k w_k h_{i-k}
        
        # Let's compute the vector A[i] = sum_k w_k h_{k-i}
        # and B[i] = sum_k w_k h_{i-k}
        
        # A is cross-correlation of w and h.
        # B is cross-correlation of w and h_rev?
        # h_{i-k} = h_rev[k-i] if h_rev[j] = h[n-1-j]? No.
        
        # Let's just do the O(N^2) loop for the gradient if it's fast enough.
        # N=600.
        # 600*600 = 360,000 multiplications.
        # 100 iterations = 36,000,000.
        # This is ~0.1s in C, ~1s in Python.
        # 10 restarts = 10s.
        # This is acceptable.
        
        # Let's implement the O(N^2) gradient for the smooth max part.
        
        for i in range(n):
            # sum_k w_k h_{k-i}
            # k-i in [0, n-1] => k in [i, i+n-1]
            # k in [0, 2n-2]
            # So k from max(0, i) to min(2n-2, i+n-1)
            k_start = max(0, i)
            k_end = min(2*n - 1, i + n - 1)
            if k_start <= k_end:
                term1 = np.dot(weights[k_start:k_end+1], h[k_start-i:k_end-i+1])
            else:
                term1 = 0.0
            
            # sum_k w_k h_{i-k}
            # i-k in [0, n-1] => k in [i-(n-1), i]
            # k in [0, 2n-2]
            # So k from max(0, i-n+1) to min(2n-2, i)
            k_start2 = max(0, i - n + 1)
            k_end2 = min(2*n - 2, i)
            if k_start2 <= k_end2:
                term2 = np.dot(weights[k_start2:k_end2+1], h[i-k_start2:i-k_end2+1][::-1])
            else:
                term2 = 0.0
            
            g_h[i] = term1 + term2
        
        # Gradient of smooth max w.r.t h
        grad_h_smooth = g_h
        
        # Total gradient w.r.t h
        grad_h = grad_h_smooth + pen_grad
        
        # Gradient w.r.t x
        grad_x = grad_h * h
        
        obj = smooth_max + penalty
        
        return obj, grad_x

    # BFGS Loop
    x_curr = x.copy()
    obj_curr, grad_curr = objective_and_grad(x_curr)
    
    for i in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # BFGS Update
        if s_prev is not None:
            # Check curvature
            rho = 1.0 / np.dot(s_prev, y_prev)
            if rho > 0:
                B = B + (1 + np.dot(y_prev, B, y_prev) * rho) * np.outer(s_prev, s_prev) - rho * (np.outer(s_prev, y_prev) + np.outer(y_prev, s_prev))
        
        # Line Search
        p = -np.linalg.solve(B, grad_curr)
        
        # Backtracking line search
        alpha_ls = 1.0
        c1 = 1e-4
        obj_new, grad_new = objective_and_grad(x_curr + alpha_ls * p)
        
        while obj_new > obj_curr + c1 * alpha_ls * np.dot(grad_curr, p):
            alpha_ls *= 0.5
            if alpha_ls < 1e-8:
                break
            x_new = x_curr + alpha_ls * p
            obj_new, grad_new = objective_and_grad(x_new)
        
        if alpha_ls < 1e-8:
            break
            
        x_new = x_curr + alpha_ls * p
        s_curr = x_new - x_curr
        y_curr = grad_new - grad_curr
        
        if np.dot(s_curr, y_curr) <= 1e-8:
            # No curvature, reset B
            B = np.eye(n)
        else:
            s_prev = s_curr
            y_prev = y_curr
        
        x_curr = x_new
        obj_curr = obj_new
        grad_curr = grad_new
        
        if np.linalg.norm(grad_curr) < 1e-5:
            break
        
    h_final = np.exp(x_curr)
    h_final = h_final / np.sum(h_final)
    return h_final

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deadline_reached():
        return time_left() < 5.0
    
    rng = np.random.default_rng(42)
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N, dtype=np.float64))
    
    # 2. Gaussian centered with various widths
    x = np.linspace(0, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        hg = np.exp(-((x - 0.5) / width) ** 2)
        candidates.append(hg)
        
    # 3. Two blocks
    for left_frac in [0.1, 0.2, 0.3, 0.4]:
        for gap_frac in [0.0, 0.05, 0.1, 0.15, 0.2]:
            left_end = int(left_frac * N)
            gap_end = left_end + int(gap_frac * N)
            if gap_end >= N:
                continue
            hb = np.zeros(N)
            hb[:left_end] = 1.0
            hb[gap_end:] = 1.0
            if np.sum(hb) > 0:
                candidates.append(hb)

    # Evaluate initial candidates
    for hc in candidates:
        if deadline_reached():
            break
        hc = np.maximum(hc, 0.0)
        if np.sum(hc) <= 0:
            continue
        sc = compute_score(hc)
        if sc < best_score:
            best_score = sc
            best_h = hc.copy()

    # Log-domain BFGS from multiple starting points
    # We use a moderate alpha for smooth max to ensure differentiability
    # and a penalty for the sum constraint
    
    starts = [
        best_h.copy(),
        np.ones(N) / N,
        np.exp(-((np.linspace(0, 1, N) - 0.5) / 0.2) ** 2),
        np.concatenate([np.ones(int(0.3*N)), np.zeros(int(0.4*N)), np.ones(int(0.3*N))]),
    ]
    
    for start_h in starts:
        if deadline_reached():
            break
        
        start_h = np.maximum(start_h, 1e-10)
        
        # Run BFGS
        h_bfgs = _log_domain_bfgs(start_h, max_iters=50, alpha=50.0, mu=10.0)
        
        sc = compute_score(h_bfgs)
        if sc < best_score:
            best_score = sc
            best_h = h_bfgs.copy()
            
        # Try with higher alpha (sharper max)
        if time_left() > 10:
            h_bfgs2 = _log_domain_bfgs(h_bfgs, max_iters=50, alpha=200.0, mu=10.0)
            sc2 = compute_score(h_bfgs2)
            if sc2 < best_score:
                best_score = sc2
                best_h = h_bfgs2.copy()

    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
