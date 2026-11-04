import numpy as np
import time
import sys

N = 600

def compute_score(h):
    # h is 1D array
    A = np.correlate(h, h, mode='full')
    # A is length 2N-1. Center is N-1.
    A0 = A[N-1]
    if A0 < 1e-10:
        return 1e10
    # Positive lags are A[N:]
    # We want max of A[1:] relative to A0
    # A[N-1] is lag 0. A[N] is lag 1.
    max_side = np.max(A[N:])
    return max_side / A0

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 20s margin
    
    best_score = 1.0
    best_h = np.ones(N)
    
    # Try several initializations
    inits = []
    
    # 1. Random
    inits.append(np.random.rand(N))
    
    # 2. Chirp-like (non-negative part of cosine chirp)
    t = np.arange(N)
    inits.append(np.maximum(0, np.cos(2 * np.pi * (t**2) / N)))
    inits.append(np.maximum(0, np.cos(2 * np.pi * (t**2) / (N/2))))
    
    # 3. Random binary
    inits.append((np.random.rand(N) > 0.5).astype(float))
    
    # 4. Gaussian windowed random
    gauss = np.exp(-0.5 * (np.arange(N) - N/2)**2 / (N/10)**2)
    inits.append(np.random.rand(N) * gauss)
    
    for h0 in inits:
        if time.time() > deadline:
            break
            
        h = h0.copy()
        # Normalize
        norm = np.linalg.norm(h)
        if norm < 1e-10:
            continue
        h = h / norm
        
        current_score = compute_score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Local optimization
        # Use a simple gradient step on the max sidelobe
        eta = 0.01
        for step in range(2000):
            if time.time() > deadline:
                break
                
            A = np.correlate(h, h, mode='full')
            A0 = A[N-1]
            if A0 < 1e-10:
                break
            
            # Get positive lags
            lags = A[N:]
            # Find max lag index (1 to N-1)
            k_max = np.argmax(lags) + 1 # lag 1 is index 0 in lags
            
            # Gradient of A(k_max) w.r.t h
            # A(k) = sum h[i] h[i+k]
            # dA/dh[i] = h[i+k] + h[i-k] (with bounds)
            
            # Vectorized gradient calculation
            # h[i+k] is np.roll(h, -k)
            # h[i-k] is np.roll(h, k)
            
            h_plus = np.roll(h, -k_max)
            h_minus = np.roll(h, k_max)
            
            # Handle boundaries:
            # For i > N-1-k, h[i+k] is 0.
            # np.roll wraps around, which is incorrect for linear correlation.
            # We need to zero out the wrapped parts.
            
            # h_plus[i] should be h[i+k_max] if i+k_max < N else 0
            # np.roll(h, -k) puts h[k] at 0.
            # We need to zero the last k elements of h_plus?
            # No, h_plus[i] = h[(i-k)%N].
            # We want h[i+k] for i < N-k.
            # So h_plus[0:N-k] = h[k:N]
            # h_plus[N-k:N] = 0
            
            h_plus = np.zeros(N)
            h_plus[:N-k_max] = h[k_max:]
            
            h_minus = np.zeros(N)
            h_minus[k_max:] = h[:N-k_max]
            
            grad = h_plus + h_minus
            
            # Update
            h = h - eta * grad
            h = np.maximum(0, h)
            
            # Renormalize
            norm = np.linalg.norm(h)
            if norm < 1e-10:
                h = h0.copy()
                norm = np.linalg.norm(h)
                h = h / norm
                continue
            h = h / norm
            
            new_score = compute_score(h)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Revert or shrink step?
                # Simple: revert
                h = h + eta * grad
                h = np.maximum(0, h)
                norm = np.linalg.norm(h)
                if norm > 1e-10:
                    h = h / norm
                else:
                    break
                eta *= 0.5
                if eta < 1e-6:
                    break
                    
    # Return the best h found
    # The task asks for a sequence of nonnegative step heights
    # The score is the upper bound on C1, which is the ratio.
    # The evaluator measures the raw value (lower is better).
    # So we should return the heights that minimize the score.
    
    return best_h.tolist()
