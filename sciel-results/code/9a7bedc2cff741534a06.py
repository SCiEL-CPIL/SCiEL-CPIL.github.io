import numpy as np
import time

N = 100
DEADLINE = time.time() + 140  # 10s margin

def _time_remaining():
    return max(0, DEADLINE - time.time())

def _compute_correlations(h):
    # Compute C_k = sum h_i h_{i+k} for all k
    # Using FFT for speed?
    # N=100 is small, direct calculation is O(N^2) = 10000 ops.
    # FFT is O(N log N).
    # Let's use FFT for safety and speed, although N is small.
    H = np.fft.fft(h)
    P = H * np.conj(H)
    C = np.real(np.fft.ifft(P))
    # Ensure real and handle numerical noise
    C = np.real(C)
    return C

def _project_to_simplex_box(h, target_sum):
    # Project h onto {x | 0 <= x_i <= 1, sum x_i = target_sum}
    # We want to find lambda such that sum(clip(h_i - lambda, 0, 1)) = target_sum
    # Let g(lambda) = sum(clip(h_i - lambda, 0, 1))
    # g is decreasing in lambda.
    # We can binary search for lambda.
    # Range of lambda: [min(h)-1, max(h)]
    
    lo = np.min(h) - 1
    hi = np.max(h)
    
    # If target_sum is outside [0, N], we can't project, but it should be N/2.
    
    for _ in range(50): # 50 iterations of binary search is plenty
        mid = (lo + hi) / 2
        x = np.clip(h - mid, 0, 1)
        s = np.sum(x)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    x = np.clip(h - (lo+hi)/2, 0, 1)
    # Adjust for any small numerical errors in sum
    # This is a very small correction
    diff = target_sum - np.sum(x)
    # Distribute diff evenly? Or add to elements not at bounds?
    # Since N is small, just adding diff/N is fine for small diff.
    x += diff / N
    # Re-clip to ensure bounds
    x = np.clip(x, 0, 1)
    # Final correction if needed (should be tiny)
    # If sum is still off, adjust the element closest to 0.5?
    # Actually, the binary search should be very accurate.
    return x

def construct():
    h = np.full(N, 0.5)
    
    best_h = h.copy()
    best_min_C = np.min(_compute_correlations(best_h))
    
    beta = 10.0  # Temperature for soft-min
    eta = 0.01   # Step size
    
    # We want to maximize min C_k.
    # Minimize L(h) = -1/beta * log(sum exp(-beta C_k))
    # Grad L = sum(w_k * grad C_k) where w_k = exp(-beta C_k) / sum(exp(-beta C_k))
    
    for step in range(2000):
        if _time_remaining() < 5:
            break
            
        C = _compute_correlations(h)
        min_C = np.min(C)
        
        if min_C > best_min_C:
            best_min_C = min_C
            best_h = h.copy()
            
        # Compute weights
        # Shift C by min_C to prevent underflow/overflow in exp
        # w_k proportional to exp(-beta * (C_k - min_C)) = exp(-beta C_k + beta min_C)
        # Since min_C is constant, it cancels in normalization.
        # So w_k proportional to exp(-beta * C_k)
        # To be safe:
        shifted_C = C - np.min(C)
        w = np.exp(-beta * shifted_C)
        w /= np.sum(w)
        
        # Compute Gradient
        # Grad C_k = roll(h, -k) + roll(h, k)
        # G = sum_k w_k * (roll(h, -k) + roll(h, k))
        # G = sum_k w_k * roll(h, -k) + sum_k w_k * roll(h, k)
        # Note: sum_k w_k * roll(h, -k) is the circular convolution of w with h?
        # Let's check.
        # (w * h)_j = sum_k w_k h_{j-k} = sum_k w_k roll(h, -k)_j ?
        # roll(h, -k)_j = h_{j-k} (if roll is defined such that roll(h, 1)_j = h_{j-1})
        # np.roll(h, k) shifts h to the right by k. So (np.roll(h, k))_j = h_{j-k}.
        # So sum_k w_k * np.roll(h, -k) is not a standard convolution directly in numpy without care.
        # Let's just compute it via FFT or loop since N=100.
        # Loop is 100 * 100 = 10000 ops. Very fast.
        
        G = np.zeros(N)
        for k in range(N):
            if w[k] < 1e-10: continue
            G += w[k] * (np.roll(h, -k) + np.roll(h, k))
            
        # Update
        h_new = h - eta * G
        
        # Project
        h = _project_to_simplex_box(h_new, N/2.0)
        
        # Adaptive step size?
        # If min_C decreased significantly, reduce eta?
        # For now, fixed eta.
        
    return best_h
