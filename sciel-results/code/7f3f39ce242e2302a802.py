import numpy as np
import time
import sys

N = 100

def _compute_overlap(h):
    # h is a 1D array of length N
    # Overlap at shift k is sum(h_i * (1 - h_{i+k}))
    # We want to minimize the max over k != 0
    # Using FFT for efficiency
    # h is real
    # 1-h is real
    # Correlation of h and 1-h
    # corr_k = sum h_i (1-h_{i+k})
    # This is the cross-correlation.
    
    # Note: The problem says "shifted 1-h".
    # Let g = 1 - h.
    # We want max_k sum h_i g_{i+k}.
    
    g = 1.0 - h
    # Cross correlation via FFT
    # We need circular correlation
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Correlation is ifft( H * conj(G) )?
    # sum h_i g_{i+k} = ifft( H * conj(G) )_k ?
    # Let's check.
    # c_k = sum_i h_i g_{i+k}
    # This is the standard cross-correlation.
    corr = np.fft.ifft(H * np.conj(G)).real
    # Exclude k=0? The prompt says "shifted", usually implies k != 0.
    # But "maximum overlap" might include k=0?
    # "shifted 1-h" usually means k is non-zero.
    # If k=0, overlap is sum h_i (1-h_i) = sum h_i - sum h_i^2.
    # For h=0.5, this is N/2 - N/4 = N/4.
    # For k!=0, it is N/4.
    # So max is N/4.
    
    # Let's assume k != 0.
    corr[0] = -np.inf
    return np.max(corr)

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    # Initial guess: 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _compute_overlap(best_h)
    
    # Try a few simple structured sequences to see if they beat 0.5
    # 1. Barker-like (random binary, then optimize)
    # 2. Sinusoidal
    
    # Optimization using L-BFGS-B
    from scipy.optimize import minimize
    
    def objective(x):
        # x is flattened h
        # Enforce sum x = N/2?
        # Minimize max overlap
        # Smooth approximation: log-sum-exp
        # Or just max
        
        # To handle the max, we can use a soft-max
        # But L-BFGS-B needs a smooth objective.
        # Let's use the raw max for now, it's non-smooth.
        # Better: minimize sum of squares of (corr_k - t) with t as auxiliary?
        # Or just use Nelder-Mead or Powell?
        
        # Given the short time, let's try a simple gradient step on the max.
        # Subgradient method.
        
        h_curr = x
        g = 1.0 - h_curr
        H = np.fft.fft(h_curr)
        G = np.fft.fft(g)
        corr = np.fft.ifft(H * np.conj(G)).real
        corr[0] = -np.inf
        max_val = np.max(corr)
        # Find the k that gives the max
        k_max = np.argmax(corr)
        
        # Gradient of corr_k w.r.t h_i
        # corr_k = sum h_i (1 - h_{i+k})
        # d/dh_i = (1 - h_{i+k}) - h_i * delta(i, i+k) ?
        # Actually, if we change h_j, it affects corr_k if j is in the sum.
        # corr_k = sum_m h_m (1 - h_{m+k})
        # d corr_k / d h_j = (1 - h_{j+k}) - h_{j-k}
        
        grad = np.zeros(N)
        # For the specific k_max
        # grad_j = (1 - h_{(j+k_max)%N}) - h_{(j-k_max)%N}
        idx1 = (np.arange(N) + k_max) % N
        idx2 = (np.arange(N) - k_max) % N
        grad = (1.0 - h_curr[idx1]) - h_curr[idx2]
        
        return max_val, grad

    # Simple gradient descent with projection
    lr = 0.01
    for i in range(500):
        if time.time() > deadline - margin:
            break
        
        val, grad = objective(h)
        h -= lr * grad
        
        # Project to sum h = N/2 and 0 <= h <= 1
        # 1. Mean correction
        h += (N/2.0 - np.sum(h)) / N
        # 2. Clip
        h = np.clip(h, 0.0, 1.0)
        # 3. Re-project to mean (clipping breaks mean)
        # Simple iteration
        for _ in range(10):
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0.0, 1.0)
        
        score = _compute_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 100

def _compute_overlap(h):
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    corr = np.fft.ifft(H * np.conj(G)).real
    corr[0] = -np.inf
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _compute_overlap(best_h)
    
    # Gradient Descent
    lr = 0.05
    for i in range(200):
        if time.time() > deadline - margin:
            break
        
        g = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        corr = np.fft.ifft(H * np.conj(G)).real
        corr[0] = -np.inf
        max_val = np.max(corr)
        k_max = np.argmax(corr)
        
        idx1 = (np.arange(N) + k_max) % N
        idx2 = (np.arange(N) - k_max) % N
        grad = (1.0 - h[idx1]) - h[idx2]
        
        h -= lr * grad
        
        # Project
        h += (N/2.0 - np.sum(h)) / N
        h = np.clip(h, 0.0, 1.0)
        for _ in range(5):
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0.0, 1.0)
            
        score = _compute_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h
# EVOLVE-BLOCK-END
