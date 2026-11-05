import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Random with non-zero sum
    np.random.seed(42)
    h = np.random.randn(N)
    h = h - np.mean(h) # Zero mean initially?
    # Actually, we want large sum.
    # Let's start with ones and add noise
    h = np.ones(N) + 0.1 * np.random.randn(N)
    
    best_h = h.copy()
    best_val = np.inf
    
    def calc_val(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.inf
        c = np.convolve(h, h)
        m = np.max(c)
        return abs(2 * N * m / (s * s))
    
    best_val = calc_val(h)
    
    # Iterative refinement
    eta = 0.01
    while time.time() < deadline:
        s = np.sum(h)
        if abs(s) < 1e-10:
            h = np.ones(N)
            s = N
            continue
            
        c = np.convolve(h, h)
        max_idx = np.argmax(c)
        max_val = c[max_idx]
        
        # Gradient of c[max_idx] w.r.t h
        # c[k] = sum h[i] h[k-i]
        # grad h[i] = h[k-i]
        # Note: k-i must be in [0, N-1]
        
        # Construct gradient
        grad = np.zeros(N)
        # k = max_idx
        # h[i] * h[k-i]
        # i ranges such that 0 <= i < N and 0 <= k-i < N
        # i >= k-N+1 and i <= k
        i_min = max(0, max_idx - N + 1)
        i_max = min(N - 1, max_idx)
        
        # grad[i] = h[max_idx - i]
        # Vectorized
        indices = np.arange(i_min, i_max + 1)
        grad[indices] = h[max_idx - indices]
        
        # We want to minimize max(c).
        # Move h in direction -grad.
        # But we must maintain sum(h) = constant?
        # If we change h, sum changes.
        # Better: minimize J(h) directly?
        # J is not smooth.
        
        # Simple update:
        # h_new = h - eta * grad
        # Then project to sum(h) = 1?
        # If we project, we add constant to all elements.
        # This might increase the peak.
        
        # Alternative: Use scipy minimize
        # But scipy might be slow.
        
        # Let's try a simple gradient step with projection
        h_new = h - eta * grad
        # Project to sum = 1
        h_new = h_new + (1.0 - np.sum(h_new)) / N
        
        val_new = calc_val(h_new)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
            eta = max(eta * 0.95, 1e-5)
        else:
            eta = max(eta * 0.5, 1e-5)
            # If stuck, try random restart or different peak?
            # For now, just reduce eta.
            
        if eta < 1e-6:
            break
            
    return best_h
