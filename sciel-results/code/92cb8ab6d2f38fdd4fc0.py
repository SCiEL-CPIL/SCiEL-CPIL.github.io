import numpy as np
import time

N = 400

def _autoconv(x):
    """Linear autoconvolution of x."""
    n = len(x)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    X = np.fft.rfft(x, n_fft)
    conv = np.fft.irfft(X * X, n_fft)
    return conv[:2*n-1]

def _c3(x):
    """C3 = max |autoconv| / N, with constraint sum(x^2)=N."""
    conv = _autoconv(x)
    return np.max(np.abs(conv)) / N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(999)
    
    # Initialize with random ±1, then scale to satisfy sum(x^2)=N
    x = rng.choice([-1.0, 1.0], size=N)
    # Already sum(x^2)=N for ±1
    
    best_x = x.copy()
    best_score = _c3(x)
    
    # Coordinate descent: adjust individual values to reduce max |autoconv|
    # while maintaining sum(x^2)=N
    
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Find the lag with max |autoconv|
        conv = _autoconv(x)
        max_lag = np.argmax(np.abs(conv))
        
        # Try to reduce the contribution at max_lag by adjusting x
        # For lag k, the autoconv term is sum_i x_i x_{i+k}
        # We can try flipping or adjusting individual x_i
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 3.0:
                break
            
            # Try flipping sign
            old = x[i]
            x[i] = -old
            new_score = _c3(x)
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = old
        
        if not improved:
            # Try small perturbations
            for i in range(N):
                if time.time() > deadline - 3.0:
                    break
                old = x[i]
                for delta in [-0.1, 0.1, -0.5, 0.5]:
                    x[i] = old + delta
                    # Rescale to maintain sum(x^2)=N
                    norm = np.sqrt(np.sum(x**2))
                    x = x * (np.sqrt(N) / norm)
                    new_score = _c3(x)
                    if new_score < best_score:
                        best_score = new_score
                        best_x = x.copy()
                        improved = True
                    else:
                        x[i] = old
                if improved:
                    break
    
    return best_x
