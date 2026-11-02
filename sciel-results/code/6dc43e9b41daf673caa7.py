import numpy as np
import time

N = 600

def compute_score(h):
    """
    Compute the normalized maximum autoconvolution score.
    Score = max_k (sum_i h_i h_{i+k}) / sum(h)
    """
    H = np.sum(h)
    if H < 1e-12:
        return np.inf
    
    # Compute autoconvolution coefficients S_k using FFT for efficiency
    # Pad to avoid circular convolution artifacts
    n_fft = 2 * N
    H_fft = np.fft.rfft(h, n=n_fft)
    S_fft = H_fft * np.conj(H_fft)
    S = np.fft.irfft(S_fft, n=n_fft)[:N]
    
    return np.max(S) / H

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initialize with uniform distribution
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Spectral shaping approach:
    # 1. Compute DFT of h
    # 2. Keep magnitudes, set phases to zero (or optimize)
    # 3. Inverse DFT
    # 4. Project onto nonnegative cone
    # 5. Repeat
    
    # We'll use an iterative algorithm:
    # - Alternate between time domain (nonnegativity) and frequency domain (peak minimization)
    
    # First, let's try a simple approach: start with various initializations
    # and use gradient descent on the smooth surrogate
    
    def smooth_max(S, beta=10.0):
        """Smooth approximation of max using log-sum-exp"""
        m = np.max(S)
        return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta
    
    def compute_S(h):
        """Compute autocorrelation S_k"""
        n_fft = 2 * N
        H_fft = np.fft.rfft(h, n=n_fft)
        S_fft = H_fft * np.conj(H_fft)
        S = np.fft.irfft(S_fft, n=n_fft)[:N]
        return S
    
    # Gradient descent with smooth surrogate
    def gradient_descent(h0, max_iter=2000, lr=0.01, beta=5.0):
        h = h0.copy()
        H = np.sum(h)
        if H < 1e-12:
            return h, np.inf
        
        # Normalize to sum = 1 for stability
        h = h / H
        
        prev_score = compute_score(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            S = compute_S(h)
            H = np.sum(h)
            
            # Smooth max objective
            m = np.max(S)
            exp_S = np.exp(beta * (S - m))
            sum_exp = np.sum(exp_S)
            obj = m + np.log(sum_exp) / beta
            
            # Gradient of smooth max of S
            # d(obj)/dh = (1/beta) * d(log sum exp)/dh + dm/dh
            # Simpler: use the weights from softmax
            weights = exp_S / sum_exp  # Shape (N,)
            
            # Gradient of S_k w.r.t. h_j
            # S_k = sum_i h_i h_{i+k}
            # dS_k/dh_j = h_{j+k} + h_{j-k} (with bounds)
            # grad_obj_j = sum_k weights_k * (h_{j+k} + h_{j-k})
            
            grad = np.zeros(N)
            for k in range(N):
                if weights[k] < 1e-10:
                    continue
                w = weights[k]
                # Contribution from h_{j+k}
                for j in range(N - k):
                    grad[j] += w * h[j + k]
                # Contribution from h_{j-k}
                for j in range(k, N):
                    grad[j] += w * h[j - k]
            
            # Project gradient onto simplex (sum h = 1, h >= 0)
            # Simple: just use gradient descent with projection
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0)
            H_new = np.sum(h_new)
            if H_new < 1e-12:
                h_new = h
            else:
                h_new = h_new / H_new
            
            score_new = compute_score(h_new)
            
            if score_new < prev_score - 1e-12:
                h = h_new
                prev_score = score_new
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return h, compute_score(h)
    
    # Try multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Gaussian
    idx = np.arange(N)
    center = (N - 1) / 2.0
    for sigma in [N/8, N/16, N/32, N/64]:
        g = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
        initializations.append(g)
    
    # 3. Triangle
    tri = np.clip(1 - np.abs(idx - center) / center, 0, 1)
    initializations.append(tri)
    
    # 4. Sharp peak (like current program)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    x_norm = x * 4.0
    base = np.maximum(0.0, 1.0 - np.abs(x_norm))
    peak = base ** 100
    initializations.append(peak)
    
    # 5. Random
    np.random.seed(42)
    initializations.append(np.random.exponential(1.0, N))
    
    # Run gradient descent from each initialization
    for h0 in initializations:
        if time.time() > deadline - 30:
            break
        
        h0 = h0 / np.sum(h0)
        h_opt, score_opt = gradient_descent(h0, max_iter=1000, lr=0.05, beta=10.0)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final polish: try a few more iterations with smaller step
    if time.time() < deadline - 10:
        h_polished, score_polished = gradient_descent(best_h, max_iter=500, lr=0.01, beta=20.0)
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
    
    # Ensure non-negativity and normalization
    best_h = np.maximum(best_h, 0)
    sum_h = np.sum(best_h)
    if sum_h > 0:
        best_h = best_h / sum_h
    
    return list(best_h)

construct()
