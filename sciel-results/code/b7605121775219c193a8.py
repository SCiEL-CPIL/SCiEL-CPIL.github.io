import numpy as np
import time

N = 50

def _ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    
    # Compute linear autocorrelation using FFT
    # Zero-pad to 2*N to avoid circular convolution artifacts
    n_fft = 2 * N
    H = np.fft.fft(h, n_fft)
    # Power spectrum
    P = H * np.conj(H)
    # Inverse FFT gives circular autocorrelation of the zero-padded sequence
    # which is equivalent to linear autocorrelation for lags 0..N-1
    corr = np.fft.ifft(P).real
    # The lags 0 to N-1 are in corr[0:N]
    peak = np.max(np.abs(corr[:N]))
    
    if peak == 0:
        return 0.0
        
    return l2_sq / (l1 * peak)

def construct():
    start_time = time.time()
    deadline = 145.0 # 150s limit, 5s margin
    
    # Initialize with a valid non-zero sequence to avoid "sum to zero" error
    # A simple Gaussian pulse is a safe starting point
    k = np.arange(N)
    init_h = np.exp(-0.5 * ((k - N/2) / 5.0)**2)
    best_h = init_h.copy()
    best_score = _ratio(best_h)
    best_params = (N/2, N/2, 5.0, 5.0, 1.0, 0.0) # c1, c2, s1, s2, a, b
    
    def evaluate_params(c1, c2, s1, s2, a, b):
        # Ensure sigma is not zero to avoid division by zero in exp
        s1 = max(s1, 0.1)
        s2 = max(s2, 0.1)
        w1 = np.exp(-0.5 * ((k - c1) / s1)**2)
        w2 = np.exp(-0.5 * ((k - c2) / s2)**2)
        h = (a * w1 + b * w2)**2
        # Ensure non-negative
        h = np.maximum(h, 0.0)
        score = _ratio(h)
        return h, score

    np.random.seed(42)
    
    # Phase 1: Random Search
    # We search over parameters for two Gaussian pulses
    # c1, c2: centers in [0, N)
    # s1, s2: widths in [0.5, 10.0]
    # a, b: amplitudes in [0.0, 1.0]
    
    max_iters = 20000
    for i in range(max_iters):
        if time.time() > deadline - 10: # Reserve 10s for polishing
            break
            
        # Sample parameters
        # Bias towards separated centers to exploit disjointness
        c1 = np.random.uniform(0, N)
        c2 = np.random.uniform(0, N)
        
        s1 = np.random.uniform(0.5, 10.0)
        s2 = np.random.uniform(0.5, 10.0)
        
        a = np.random.uniform(0.0, 1.0)
        b = np.random.uniform(0.0, 1.0)
        
        if a == 0 and b == 0:
            continue
            
        h, score = evaluate_params(c1, c2, s1, s2, a, b)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
            best_params = (c1, c2, s1, s2, a, b)

    # Phase 2: Local Polish
    # Refine the best parameters found
    if time.time() < deadline - 5:
        c1, c2, s1, s2, a, b = best_params
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005]
        
        for step in step_sizes:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Perturb each parameter
                for i in range(6):
                    for sign in [-1, 1]:
                        new_params = list(best_params)
                        # Scale step size relative to parameter magnitude or default
                        param_val = best_params[i]
                        scale = max(1.0, abs(param_val))
                        new_params[i] = param_val + sign * step * scale
                        
                        # Clamp parameters to valid ranges
                        new_params[0] = np.clip(new_params[0], 0, N) # c1
                        new_params[1] = np.clip(new_params[1], 0, N) # c2
                        new_params[2] = np.clip(new_params[2], 0.1, 20.0) # s1
                        new_params[3] = np.clip(new_params[3], 0.1, 20.0) # s2
                        new_params[4] = np.clip(new_params[4], 0, 2.0) # a
                        new_params[5] = np.clip(new_params[5], 0, 2.0) # b
                        
                        h_new, score_new = evaluate_params(*new_params)
                        if score_new > best_score:
                            best_score = score_new
                            best_h = h_new.copy()
                            best_params = tuple(new_params)
                            improved = True

    # Final Safety Check: Ensure best_h is not all zeros
    if np.all(best_h == 0):
        best_h = init_h
        
    return best_h.tolist()
