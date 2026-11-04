import numpy as np

N = 400

def construct():
    best_h = np.ones(N)
    best_val = np.inf
    
    # Try several random initializations
    for _ in range(10):
        h = np.random.randn(N)
        # Ensure sum is not zero
        if np.sum(h) == 0:
            h[0] = 1
        h = h / np.sum(h)
        
        # Gradient Descent with Subgradient
        for step in range(200):
            C = np.convolve(h, h)
            M = np.max(C)
            k_max = np.argmax(C)
            S = np.sum(h)
            
            # Gradient of M with respect to h
            # C[k] = sum_{i+j=k} h[i]h[j]
            # dC[k_max]/dh[j] = h[k_max - j]
            # Note: indices must be valid
            idx = k_max - np.arange(N)
            valid = (idx >= 0) & (idx < N)
            grad_M = np.zeros(N)
            grad_M[valid] = h[idx[valid]]
            
            # Gradient of Value = M / S^2
            # dV/dh = (1/S^2) * dM/dh - (2M/S^3) * dS/dh
            # dS/dh = 1
            grad_V = (1.0 / (S**2)) * grad_M - (2.0 * M / (S**3)) * np.ones(N)
            
            # Step
            eta = 0.01
            h = h - eta * grad_V
            
            # Normalize sum to 1
            h = h / np.sum(h)
            
        # Evaluate final
        C = np.convolve(h, h)
        M = np.max(C)
        S = np.sum(h)
        val = abs(2 * N * M / (S**2))
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
