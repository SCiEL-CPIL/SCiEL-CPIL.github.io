import numpy as np

N = 100

def _compute_C(x):
    C = np.zeros(N - 1)
    for k in range(1, N):
        C[k - 1] = np.dot(x[:N - k], x[k:])
    return C

def _project(x):
    # Project onto sum x = N/2 and 0 <= x <= 1
    # This is a standard isotonic-like projection or water-filling
    # Simple iterative clipping
    for _ in range(50):
        x = np.clip(x, 0.0, 1.0)
        diff = N / 2.0 - np.sum(x)
        if abs(diff) < 1e-10:
            break
        # Adjust non-saturated elements
        non_sat = (x > 1e-6) & (x < 1.0 - 1e-6)
        if np.sum(non_sat) == 0:
            x += diff / N
            x = np.clip(x, 0.0, 1.0)
        else:
            x[non_sat] += diff / np.sum(non_sat)
    return x

def construct():
    # Initialize with a sinusoidal pattern to break symmetry
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 0.5 + 0.2 * np.sin(t)
    x = _project(x)
    
    # Subgradient ascent on min C_k
    # We want to maximize min(C).
    # f(x) = min_k C_k(x).
    # Gradient of min is gradient of the argmin.
    
    best_x = x.copy()
    best_min_C = np.min(_compute_C(x))
    
    alpha = 0.01
    for step in range(1000):
        C = _compute_C(x)
        k_min = np.argmin(C) + 1
        current_min = C[k_min - 1]
        
        # Gradient of C_{k_min}
        grad = np.zeros(N)
        # d C_k / d x_j = x_{j+k} (if j < N-k) + x_{j-k} (if j >= k)
        idx1 = np.arange(N - k_min)
        grad[idx1] += x[idx1 + k_min]
        idx2 = np.arange(k_min, N)
        grad[idx2] += x[idx2 - k_min]
        
        # Update
        x_new = x + alpha * grad
        x_new = _project(x_new)
        
        # Check improvement
        C_new = _compute_C(x_new)
        min_new = np.min(C_new)
        
        if min_new > best_min_C:
            best_min_C = min_new
            best_x = x_new.copy()
        
        x = x_new
        
        # Decay alpha
        alpha *= 0.999
        
    return best_x
