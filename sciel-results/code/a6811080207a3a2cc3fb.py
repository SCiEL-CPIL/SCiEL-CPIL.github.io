import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
DEADLINE = 145  # Safety margin from 150s limit
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _autoconvolution_fft(h):
    """Compute autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_padded = np.zeros(size)
    h_padded[:n] = h
    fft_h = np.fft.fft(h_padded)
    conv = np.fft.ifft(fft_h * fft_h).real
    return conv[:2 * n - 1]

def _score(h):
    """Compute the ratio: (sum h^2) / (max autoconvolution) for normalized h."""
    h = np.array(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0, None)
    total = np.sum(h)
    if total == 0:
        return 0.0
    h_norm = h / total
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconvolution_fft(h_norm)
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / max_conv

def _gradient_flow(h_init, max_iter=5000, eta0=0.05, decay=0.9999):
    """
    Gradient ascent on log(score) using multiplicative updates.
    Maximizes (sum h^2) / max(h*h) subject to h>=0, sum(h)=1.
    """
    h = h_init.copy()
    h = np.clip(h, 1e-15, None)
    h = h / np.sum(h)
    
    best_h = h.copy()
    best_score = _score(h)
    eta = eta0
    
    for it in range(max_iter):
        if _time_remaining() < 5:
            break
        
        # Compute autoconvolution and peak
        conv = _autoconvolution_fft(h)
        max_conv = np.max(conv)
        
        if max_conv <= 0:
            break
            
        # Identify peak locations (active set)
        # Use a small tolerance to include near-peaks for smoothing
        tol = 1e-8 * max_conv
        active = np.where(np.abs(conv - max_conv) < tol)[0]
        
        # Gradient of M = max(conv) w.r.t h
        # M = sum_{m in active} conv_m (approximately, if we treat as sum)
        # d conv_m / d h_i = h_{m-i} + h_{i-m}
        # This is 2 * (h * indicator_active)(i) roughly?
        # Let's compute grad_M efficiently.
        # grad_M_i = sum_{m in active} (h_{m-i} + h_{i-m})
        
        # We can compute this via FFT or direct summation if active set is small.
        # Active set is usually small (1-3 indices).
        grad_M = np.zeros(N)
        for m in active:
            # h_{m-i} is valid for i in [0, N-1] and m-i in [0, N-1]
            # h_{i-m} is valid for i in [0, N-1] and i-m in [0, N-1]
            
            # Vectorized:
            # Indices i such that 0 <= m-i < N => m-N < i <= m
            # Indices i such that 0 <= i-m < N => m < i <= m+N-1
            
            # Term 1: h[m-i]
            # i ranges from max(0, m-(N-1)) to min(N-1, m)
            i_start1 = max(0, m - (N - 1))
            i_end1 = min(N - 1, m)
            if i_start1 <= i_end1:
                idxs1 = np.arange(i_start1, i_end1 + 1)
                vals1 = h[m - idxs1]
                grad_M[idxs1] += vals1
            
            # Term 2: h[i-m]
            # i ranges from max(0, m) to min(N-1, m+N-1)
            i_start2 = max(0, m)
            i_end2 = min(N - 1, m + N - 1)
            if i_start2 <= i_end2:
                idxs2 = np.arange(i_start2, i_end2 + 1)
                vals2 = h[idxs2 - m]
                grad_M[idxs2] += vals2
        
        # Gradient of log(score)
        # score = l2_sq / M
        # log(score) = log(l2_sq) - log(M)
        # d log(l2_sq) / d h = 2 * h / l2_sq
        # d log(M) / d h = grad_M / M
        
        l2_sq = np.sum(h**2)
        grad_log_score = (2 * h / l2_sq) - (grad_M / max_conv)
        
        # Multiplicative update: h_new = h * exp(eta * grad_log_score)
        # To maintain stability, clip the exponent
        exp_arg = np.clip(eta * grad_log_score, -10, 10)
        h_new = h * np.exp(exp_arg)
        
        # Normalize and clip
        h_new = np.clip(h_new, 0, None)
        sum_h = np.sum(h_new)
        if sum_h == 0:
            break
        h_new = h_new / sum_h
        
        # Check improvement
        s_new = _score(h_new)
        if s_new > best_score:
            best_score = s_new
            best_h = h_new.copy()
        
        h = h_new
        eta *= decay
        
    return best_h, best_score

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Initial guesses
    # 1. Uniform
    h0 = np.ones(N) / N
    s0 = _score(h0)
    if s0 > best_score:
        best_score = s0
        best_h = h0.copy()
        
    # 2. Peak at 0
    h1 = np.zeros(N)
    h1[0] = 1.0
    s1 = _score(h1)
    if s1 > best_score:
        best_score = s1
        best_h = h1.copy()
        
    # 3. Two peaks
    h2 = np.zeros(N)
    h2[0] = 0.5
    h2[1] = 0.5
    s2 = _score(h2)
    if s2 > best_score:
        best_score = s2
        best_h = h2.copy()

    # 4. Random sparse
    np.random.seed(42)
    for _ in range(5):
        support = np.random.choice(N, np.random.randint(2, 10), replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / len(support)
        s_rand = _score(h_rand)
        if s_rand > best_score:
            best_score = s_rand
            best_h = h_rand.copy()

    # Run Gradient Flow from multiple starting points
    # We will run it from the current best and a few other diverse points
    start_points = [
        best_h.copy(),
        np.ones(N) / N,
        np.zeros(N)
    ]
    
    # Add a few random starts
    for _ in range(3):
        h_rand = np.random.rand(N)
        h_rand /= np.sum(h_rand)
        start_points.append(h_rand)
        
    # Also try supports based on symmetry
    # Symmetric around N/2?
    h_sym = np.zeros(N)
    h_sym[N//2] = 1.0
    start_points.append(h_sym)
    
    for h_init in start_points:
        if _time_remaining() < 10:
            break
            
        h_opt, s_opt = _gradient_flow(h_init, max_iter=2000, eta0=0.1, decay=0.9995)
        
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
    # Final polish with a very small step size gradient flow
    if _time_remaining() > 10:
        h_final, s_final = _gradient_flow(best_h, max_iter=1000, eta0=0.01, decay=0.9999)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
