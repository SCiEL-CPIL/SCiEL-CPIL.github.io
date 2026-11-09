import numpy as np
import time

N = 600

def _metric_fast(h):
    h = np.maximum(h, 0.0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_h = np.ones(N)
    best_score = _metric_fast(best_h)
    
    # Try structured initial guesses
    guesses = [
        np.ones(N),
        np.sin(np.linspace(0, np.pi, N)) + 0.01,
        np.exp(-((np.arange(N) - N/2)**2) / (2*(N/3)**2)),
        np.concatenate([np.ones(N//2), np.zeros(N//2)]),
        np.concatenate([np.ones(N//3), np.zeros(N//3), np.ones(N//3)]),
        1.0 + 0.5*np.sin(2*np.pi*np.arange(N)/N),
        1.0 + 0.5*np.sin(4*np.pi*np.arange(N)/N),
        np.linspace(0.5, 1.5, N),
    ]
    
    for g in guesses:
        g = np.maximum(g, 0.0)
        s = _metric_fast(g)
        if s < best_score:
            best_score = s
            best_h = g.copy()
    
    # Multiple hill climbing restarts
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            h = best_h.copy()
        else:
            h = best_h.copy()
            h += np.random.randn(N) * 0.1 * max(1.0, np.mean(h))
            h = np.maximum(h, 0.0)
        
        current_score = _metric_fast(h)
        step_size = 0.1 * max(1.0, np.mean(h))
        
        for iteration in range(50000):
            if time.time() > deadline - 5:
                break
            
            idx = np.random.randint(0, N)
            old_val = h[idx]
            delta = np.random.randn() * step_size
            h[idx] = max(0.0, old_val + delta)
            
            new_score = _metric_fast(h)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                step_size *= 0.999
            else:
                h[idx] = old_val
                step_size = min(step_size * 1.001, 0.5 * max(1.0, np.mean(h)))
    
    return [float(v) for v in best_h]
