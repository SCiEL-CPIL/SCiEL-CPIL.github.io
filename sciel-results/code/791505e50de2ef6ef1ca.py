"""Program using model-and-optimize approach for first autocorrelation inequality."""
import time
import numpy as np
from scipy import optimize

N = 600

def _fft_convolve(h):
    """Fast linear convolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, fft_size)
    conv = np.fft.irfft(H * H, fft_size)[:size]
    return conv

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_smooth(h, beta=100.0):
    """
    Smooth surrogate for compute_score using log-sum-exp.
    This is differentiable and suitable for gradient-based optimization.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    # Log-sum-exp for smooth max
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    # Smooth max: (1/beta) * log(sum(exp(beta * conv / max_val))) * max_val / s^2
    # Simpler: use softplus-like approximation
    shifted = conv - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted * beta))) / beta
    return log_sum_exp / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # We want to minimize max_k conv[k] / (sum h)^2 subject to h[i] >= 0.
    # This is scale invariant. We can fix sum(h) = 1.
    # The problem is equivalent to finding a probability mass function p on {0,...,N-1}
    # that minimizes the peak of its linear convolution (p * p).
    # Note: linear convolution of length N vectors has length 2N-1.
    
    # Coordinates x_i:
    # The intervals are equal width on [-1/4, 1/4].
    # Width w = 0.5 / 600 = 1/1200.
    # Center of interval i: x_i = -0.25 + (i + 0.5) * w.
    
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)
    
    # Family 1: h(x) = exp(-(|x| / sigma)^p)
    def param_h1(sigma, p):
        if sigma <= 0 or p <= 0:
            return np.ones(N) / N
        h = np.exp(-np.abs(x / sigma) ** p)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Family 2: h(x) = (1 - |x|/L)^a for |x| < L, 0 otherwise
    # L controls support width (fraction of half-interval)
    def param_h2(a, L_frac):
        if a <= 0 or L_frac <= 0:
            return np.ones(N) / N
        L = L_frac * 0.25  # L_frac in (0, 1], L in (0, 0.25]
        abs_x = np.abs(x)
        h = np.zeros(N)
        mask = abs_x < L
        h[mask] = (1 - abs_x[mask] / L) ** a
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Family 3: h(x) = (1 - (|x|/L)^a)^b for |x| < L, 0 otherwise
    def param_h3(a, b, L_frac):
        if a <= 0 or b <= 0 or L_frac <= 0:
            return np.ones(N) / N
        L = L_frac * 0.25
        abs_x = np.abs(x)
        h = np.zeros(N)
        mask = abs_x < L
        t = (abs_x[mask] / L) ** a
        h[mask] = (1 - t) ** b
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Smooth surrogate objective using log-sum-exp
    def smooth_objective(h, beta=50.0):
        s = np.sum(h)
        if s == 0:
            return 0.0
        conv = _fft_convolve(h)
        max_val = np.max(conv)
        if max_val <= 0:
            return 0.0
        shifted = conv - max_val
        # Use a stable log-sum-exp
        log_sum = max_val + np.log(np.sum(np.exp(shifted * beta))) / beta
        return log_sum / (s * s)
    
    # Initialize best
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Grid search over family 1
    sigmas = [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]
    ps = [0.5, 1, 1.5, 2, 3, 4, 6]
    
    for s0 in sigmas:
        for p0 in ps:
            if time.time() > deadline:
                break
            h0 = param_h1(s0, p0)
            score0 = compute_score(h0)
            if score0 < best_score:
                best_score = score0
                best_h = h0.copy()
    
    # Grid search over family 2
    a_vals2 = [0.5, 1, 1.5, 2, 3, 5]
    L_fracs = [0.3, 0.5, 0.7, 0.9, 1.0]
    
    for a0 in a_vals2:
        for L0 in L_fracs:
            if time.time() > deadline:
                break
            h0 = param_h2(a0, L0)
            score0 = compute_score(h0)
            if score0 < best_score:
                best_score = score0
                best_h = h0.copy()
    
    # Grid search over family 3
    a_vals3 = [0.5, 1, 2, 3, 4]
    b_vals3 = [0.5, 1, 2, 3, 4, 6]
    L_fracs3 = [0.5, 0.7, 0.9, 1.0]
    
    for a0 in a_vals3:
        for b0 in b_vals3:
            for L0 in L_fracs3:
                if time.time() > deadline:
                    break
                h0 = param_h3(a0, b0, L0)
                score0 = compute_score(h0)
                if score0 < best_score:
                    best_score = score0
                    best_h = h0.copy()
    
    # Local optimization using smooth surrogate
    # Family 1 optimization
    def obj1_smooth(params):
        sigma, p = params
        sigma = np.clip(sigma, 0.01, 0.5)
        p = np.clip(p, 0.3, 10.0)
        h = param_h1(sigma, p)
        return smooth_objective(h, beta=50.0)
    
    # Family 2 optimization
    def obj2_smooth(params):
        a, L_frac = params
        a = np.clip(a, 0.1, 10.0)
        L_frac = np.clip(L_frac, 0.1, 1.0)
        h = param_h2(a, L_frac)
        return smooth_objective(h, beta=50.0)
    
    # Family 3 optimization
    def obj3_smooth(params):
        a, b, L_frac = params
        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, 0.1, 10.0)
        L_frac = np.clip(L_frac, 0.1, 1.0)
        h = param_h3(a, b, L_frac)
        return smooth_objective(h, beta=50.0)
    
    # Optimize family 1 from best grid points
    max_opts = 15
    opt_count = 0
    
    for s0 in sigmas:
        for p0 in ps:
            if time.time() > deadline or opt_count >= max_opts:
                break
            opt_count += 1
            bounds = [(0.01, 0.5), (0.3, 10.0)]
            try:
                res = optimize.minimize(
                    obj1_smooth, 
                    [s0, p0], 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-9}
                )
                if res.fun < best_score:
                    h_candidate = param_h1(res.x[0], res.x[1])
                    actual_score = compute_score(h_candidate)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_candidate
            except Exception:
                pass
    
    # Optimize family 2
    for a0 in a_vals2:
        for L0 in L_fracs:
            if time.time() > deadline or opt_count >= max_opts + 10:
                break
            opt_count += 1
            bounds = [(0.1, 10.0), (0.1, 1.0)]
            try:
                res = optimize.minimize(
                    obj2_smooth, 
                    [a0, L0], 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-9}
                )
                if res.fun < best_score:
                    h_candidate = param_h2(res.x[0], res.x[1])
                    actual_score = compute_score(h_candidate)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_candidate
            except Exception:
                pass
    
    # Optimize family 3
    for a0 in a_vals3:
        for b0 in b_vals3:
            for L0 in L_fracs3:
                if time.time() > deadline or opt_count >= max_opts + 20:
                    break
                opt_count += 1
                bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 1.0)]
                try:
                    res = optimize.minimize(
                        obj3_smooth, 
                        [a0, b0, L0], 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 20, 'ftol': 1e-9}
                    )
                    if res.fun < best_score:
                        h_candidate = param_h3(res.x[0], res.x[1], res.x[2])
                        actual_score = compute_score(h_candidate)
                        if actual_score < best_score:
                            best_score = actual_score
                            best_h = h_candidate
                except Exception:
                    pass
    
    # Final polish: Direct optimization on coefficients using softmax parameterization
    if time.time() < deadline - 5:
        # Convert best_h to theta (log-space)
        # Ensure h is strictly positive for log
        h_init = np.maximum(best_h, 1e-12)
        h_init = h_init / np.sum(h_init)
        theta_init = np.log(h_init)
        
        beta = 50.0 # Sharpness of smooth max
        
        def softmax_obj_jac(theta):
            # Compute h from theta
            theta_shifted = theta - np.max(theta)
            exp_theta = np.exp(theta_shifted)
            h = exp_theta / np.sum(exp_theta)
            
            # Compute convolution
            conv = _fft_convolve(h)
            
            # Smooth max
            max_val = np.max(conv)
            if max_val <= 0:
                return 0.0, np.zeros_like(theta)
            
            shifted = conv - max_val
            # Weights for the smooth max
            # d(log_sum_exp)/d(conv_k) = exp(beta * conv_k) / sum(exp(beta * conv_m))
            # To avoid overflow, we work with shifted
            exp_shifted = np.exp(shifted * beta)
            sum_exp = np.sum(exp_shifted)
            if sum_exp == 0:
                return 0.0, np.zeros_like(theta)
            weights = exp_shifted / sum_exp
            
            # Objective value
            val = (max_val + np.log(sum_exp) / beta) / (np.sum(h) ** 2)
            # Note: sum(h) is 1 by construction, so val is just the smooth max
            
            # Gradient w.r.t h
            # d(val)/d(conv_k) = weights[k] * beta? No.
            # val = (1/beta) * log(sum(exp(beta * conv)))
            # d(val)/d(conv_k) = exp(beta * conv_k) / sum(exp(beta * conv_m)) = weights[k]
            # So grad_conv = weights
            
            # d(conv_k)/d(h_i)
            # conv_k = sum_{a+b=k} h_a h_b
            # d(conv_k)/d(h_i) = sum_{b: i+b=k} h_b + sum_{a: a+i=k} h_a = 2 * sum_{b: i+b=k} h_b
            # Let's compute this via convolution.
            # Let g_i = sum_{k} d(val)/d(conv_k) * d(conv_k)/d(h_i)
            # g_i = sum_k weights[k] * 2 * h_{k-i} (with bounds)
            # This is 2 * (weights * h) (linear convolution, taking the part where index is in 0..N-1)
            # Actually, h_{k-i} is non-zero only if 0 <= k-i < N.
            # So g = 2 * (weights * h)_{0:N} ?
            # Let's check:
            # (w * h)_i = sum_j w_j h_{i-j}.
            # We need sum_k w_k h_{k-i}. Let j = k-i => k = i+j.
            # sum_j w_{i+j} h_j.
            # This is cross-correlation or convolution with reversed weights.
            # Let's just compute it explicitly if N is small, or use FFT.
            # N=600. Explicit is O(N^2) = 360k ops. Fast enough.
            # But we are in a loop. Let's use FFT for gradient too.
            
            # grad_h_i = 2 * sum_{k: i <= k < i+N} weights[k] * h[k-i]
            # This is the correlation of weights and h.
            # corr[i] = sum_j weights[i+j] h[j]
            
            # Using FFT:
            # corr = ifft(fft(weights) * conj(fft(h)))? 
            # Standard cross-correlation R_xy[n] = sum_m x[m+n] y[m].
            # Here x=weights, y=h.
            # We can compute via FFT.
            
            # Simpler: just compute the gradient of the objective w.r.t h directly using the definition.
            # grad_h = 2 * (weights * h_reversed) ?
            
            # Let's stick to the analytical derivation:
            # grad_h_i = 2 * sum_{j=0}^{N-1} h[j] * weights[i+j]
            
            # This is the convolution of h and weights, but reversed?
            # Let's just do the O(N^2) loop if it's fast enough for 600.
            # 600*600 = 360,000 additions. In Python, this is slow (~0.1s).
            # In NumPy, we can use stride tricks or just FFT.
            
            # FFT approach for cross-correlation:
            # corr = np.fft.irfft(np.fft.rfft(weights, 2*N) * np.conj(np.fft.rfft(h, 2*N)))
            # The correlation of a and b is ifft(fft(a) * conj(fft(b))).
            # We want sum_j h[j] weights[i+j].
            # Let's verify with small example.
            # h=[h0, h1], w=[w0, w1, w2]
            # i=0: h0*w0 + h1*w1
            # i=1: h0*w1 + h1*w2
            # This is cross-correlation.
            
            size = 2 * N
            H = np.fft.rfft(h, size)
            W = np.fft.rfft(weights, size)
            # Cross correlation r[i] = sum_j h[j] w[i+j]
            # This is ifft(conj(H) * W) ?
            # Let's check:
            # ifft(conj(H) W) = sum_k conj(H_k) W_k e^{i 2 pi k n / size}
            # H_k = sum_j h_j e^{-i 2 pi k j / size}
            # conj(H_k) = sum_j h_j e^{i 2 pi k j / size}
            # W_k = sum_m w_m e^{-i 2 pi k m / size}
            # Product = sum_j sum_m h_j w_m e^{i 2 pi k (j-m) / size}
            # ifft -> sum_k ... e^{i 2 pi k n / size} = delta_{n, m-j}
            # So result[n] = sum_j h_j w_{n+j}.
            # Yes!
            
            grad_h = 2.0 * np.fft.irfft(np.conj(H) * W, size)[:N]
            
            # Gradient w.r.t theta
            # h_i = exp(theta_i) / sum(exp(theta))
            # d h_i / d theta_j = h_i (delta_ij - h_j)
            # grad_theta = grad_h * h * (1 - h) ? No.
            # sum_i grad_h_i d h_i / d theta_j
            # = sum_i grad_h_i h_i (delta_ij - h_j)
            # = grad_h_j h_j - h_j sum_i grad_h_i h_i
            # Let S = sum_i grad_h_i h_i
            # grad_theta_j = h_j (grad_h_j - S)
            
            S = np.dot(grad_h, h)
            grad_theta = h * (grad_h - S)
            
            return val, grad_theta
        
        try:
            # Use L-BFGS-B
            # We don't have bounds on theta, but softmax handles it.
            res = optimize.minimize(
                softmax_obj_jac, 
                theta_init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_score:
                # Get h from res.x
                theta_opt = res.x
                theta_opt_shifted = theta_opt - np.max(theta_opt)
                h_opt = np.exp(theta_opt_shifted)
                h_opt = h_opt / np.sum(h_opt)
                
                actual_score = compute_score(h_opt)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_opt
        except Exception:
            pass

    # Final check: ensure best_h is valid (non-negative)
    best_h = np.maximum(0, best_h)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
