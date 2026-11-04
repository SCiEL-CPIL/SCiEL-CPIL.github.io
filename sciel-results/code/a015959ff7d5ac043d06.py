"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Objective: minimize max_k sum_i h[i] * (1 - h[(i-k) % N])
    # This is the max over shifts of the cross-correlation of h with 1-h
    
    def overlap_max(h):
        # h has length N, values in [0,1], sum = N/2
        # Overlap at shift k: sum_i h[i] * (1 - h[(i-k) % N])
        # = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
        # = N/2 - autocorr(h, k)
        # So minimizing max overlap = maximizing min autocorr sidelobe
        # autocorr at lag 0 is sum h[i]^2
        # We want to maximize the minimum of autocorr(k) for k != 0
        # subject to sum h = N/2, 0 <= h <= 1
        
        # Compute all circular autocorrelations
        # autocorr(k) = sum_i h[i] * h[(i+k) % N]
        H = np.fft.fft(h)
        acf = np.fft.ifft(np.abs(H)**2).real
        # acf[k] = sum_i h[i] * h[(i+k)%N]
        # overlap at shift k = N/2 - acf[k]
        # We want to minimize max_k (N/2 - acf[k]) for k != 0
        # = N/2 - min_{k!=0} acf[k]
        # So we want to maximize min_{k!=0} acf[k]
        
        # Exclude lag 0
        acf_sidelobes = np.concatenate([acf[1:], acf[:0]])  # just acf[1:] for circular
        # Actually for circular autocorr, lags 1..N-1
        min_sidelobe = np.min(acf[1:])
        return N/2 - min_sidelobe
    
    def project(h):
        # Project onto {h: 0 <= h[i] <= 1, sum h = N/2}
        # Simple: clip then adjust
        h = np.clip(h, 0, 1)
        diff = N/2 - np.sum(h)
        if abs(diff) > 1e-10:
            # Distribute uniformly
            h += diff / N
            h = np.clip(h, 0, 1)
            # If still off, do a simple water-filling
            for _ in range(10):
                diff = N/2 - np.sum(h)
                if abs(diff) < 1e-12:
                    break
                if diff > 0:
                    # Need to add, add to those below 1
                    mask = h < 1
                    if np.sum(mask) == 0:
                        break
                    h[mask] += diff / np.sum(mask)
                else:
                    mask = h > 0
                    if np.sum(mask) == 0:
                        break
                    h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        return h
    
    # Initialize with a Zadoff-Chu-like or m-sequence pattern
    # Use a simple binary pattern with good autocorrelation
    # Start with alternating pattern (not great) or use a known good sequence
    
    # Try: use the first N terms of a maximum-length sequence (m-sequence)
    # For N=100, we can use a sinusoidal pattern or random start
    
    # Better: start with a "balanced" random binary sequence
    np.random.seed(42)
    best_h = None
    best_score = float('inf')
    
    # Multiple random restarts with gradient descent
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Initialize: random binary-like, then smooth
        if restart == 0:
            # Alternating
            h0 = np.where(np.arange(N) % 2 == 0, 0.8, 0.2)
        elif restart == 1:
            # Sinusoidal
            h0 = 0.5 + 0.4 * np.sin(2 * np.pi * 3 * np.arange(N) / N)
        elif restart == 2:
            # Another sinusoid
            h0 = 0.5 + 0.4 * np.sin(2 * np.pi * 7 * np.arange(N) / N)
        else:
            # Random
            h0 = np.random.uniform(0, 1, N)
        
        h = project(h0.copy())
        
        # Gradient descent with projected gradient
        lr = 0.1
        for step in range(500):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient of overlap_max w.r.t. h
            # overlap_max = N/2 - min_{k!=0} acf[k]
            # This is non-smooth, so use subgradient:
            # d(overlap_max)/d(h) = -d(min acf)/d(h)
            # min acf is achieved at some k*, so gradient = -d(acf[k*])/d(h)
            
            H = np.fft.fft(h)
            acf = np.fft.ifft(np.abs(H)**2).real
            k_star = np.argmin(acf[1:]) + 1
            k_star = k_star % N
            
            # acf[k] = sum_i h[i] * h[(i+k)%N]
            # d(acf[k])/d(h[j]) = h[(j+k)%N] + h[(j-k)%N]
            grad_acf = h[(np.arange(N) + k_star) % N] + h[(np.arange(N) - k_star) % N]
            
            # We want to maximize min acf, so gradient of objective (to minimize) is -grad_acf
            grad = -grad_acf
            
            # Projected gradient step
            h_new = h - lr * grad
            h_new = project(h_new)
            
            # Check convergence
            h = h_new
        
        # Evaluate
        score = overlap_max(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        # Fallback: uniform
        best_h = np.full(N, 0.5)
    
    # Final projection to ensure exact sum
    best_h = project(best_h)
    
    # Verify sum is exactly N/2
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h

# EVOLVE-BLOCK-END
