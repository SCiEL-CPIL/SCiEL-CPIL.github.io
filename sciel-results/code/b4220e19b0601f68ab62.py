import numpy as np

N = 400

def construct():
    best_h = np.ones(N)
    best_val = np.inf
    
    # Try several random initializations
    for _ in range(50):
        # Random initialization
        h = np.random.randn(N)
        
        # Ensure sum is not zero and normalize to sum=1
        S = np.sum(h)
        if np.abs(S) < 1e-9:
            h[0] = 1
            S = np.sum(h)
        h = h / S
        
        # Optimize
        for step in range(500):
            # Compute autocorrelation
            C = np.convolve(h, h)
            k_max = np.argmax(C)
            M = C[k_max]
            
            # Calculate gradient of M with respect to h
            # dM/dh[j] = 2 * h[k_max - j] (if valid)
            idx = k_max - np.arange(N)
            valid = (idx >= 0) & (idx < N)
            grad = np.zeros(N)
            grad[valid] = 2.0 * h[idx[valid]]
            
            # Project gradient to sum 0 to maintain constraint sum(h)=1
            grad = grad - np.mean(grad)
            
            # Step size
            eta = 0.05
            
            h_new = h - eta * grad
            
            # Renormalize
            S_new = np.sum(h_new)
            if np.abs(S_new) < 1e-9:
                break
            h_new = h_new / S_new
            
            h = h_new
            
        # Evaluate
        C = np.convolve(h, h)
        M = np.max(C)
        val = abs(2 * N * M) # Since sum(h)=1
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
