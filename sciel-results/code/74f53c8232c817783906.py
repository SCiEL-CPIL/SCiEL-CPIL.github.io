import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Baseline: All ones
    x_best = np.ones(N, dtype=np.float64)
    best_score = _score(x_best)
    
    # Try to improve using gradient descent on continuous values
    # We fix sum(x) = 1 to remove scale invariance
    # Minimize max(|ac|)
    
    x = np.ones(N, dtype=np.float64)
    x /= np.sum(x) # sum is 1
    
    rng = np.random.default_rng(42)
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
        
    # Gradient Descent
    # We want to minimize f(x) = max_k |ac[k]|
    # subject to sum(x) = 1
    
    # We use a simple subgradient method
    # If ac[k_max] > 0, we want to decrease ac[k_max]
    # If ac[k_max] < 0, we want to increase ac[k_max]
    
    step_size = 0.01
    no_improve = 0
    max_no_improve = 50
    
    for iter in range(5000):
        if time.time() > deadline - 10:
            break
            
        ac = _autoconv(x)
        abs_ac = np.abs(ac)
        k_max = np.argmax(abs_ac)
        val_max = ac[k_max]
        
        # Current score
        s = np.sum(x)
        cur_score = abs(val_max) / (s*s)
        
        if cur_score < best_score:
            best_score = cur_score
            x_best = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
                
        # Compute gradient of ac[k_max] w.r.t x
        # ac[k] = sum_i x[i] x[k-i]
        # d(ac[k])/dx[j] = x[k-j] + x[j-k]
        
        grad = np.zeros(N)
        # k_max is in range [0, 2N-2]
        k = k_max
        for j in range(N):
            # term 1: x[k-j] if 0 <= k-j < N
            if 0 <= k - j < N:
                grad[j] += x[k - j]
            # term 2: x[j-k] if 0 <= j-k < N
            if 0 <= j - k < N:
                grad[j] += x[j - k]
                
        # We want to decrease |ac[k]|
        # If val_max > 0, decrease ac[k] -> move x in -grad direction
        # If val_max < 0, increase ac[k] -> move x in +grad direction
        direction = -np.sign(val_max)
        
        # Update x
        x_new = x + step_size * direction * grad
        
        # Enforce sum(x_new) = 1
        x_new = x_new - np.mean(x_new) + 1.0/N
        
        # Optional: Enforce positivity? "Signed" allows negative.
        # But if we go negative, sum might drop.
        # Let's allow negative values.
        
        x = x_new
        
    # Final check
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score

    return x_best
