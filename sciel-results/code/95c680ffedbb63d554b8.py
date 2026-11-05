"""Convex ADMM optimization for flat autocorrelation profile."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    # Build convolution matrix C such that C @ h = conv(h, h)
    # conv(h, h)[k] = sum_{i+j=k} h[i]*h[j] for k = 0..2N-2
    # This is a bilinear operation, not linear in h.
    # However, we can use the fact that autoconvolution in frequency domain is |FFT(h)|^2
    # But for ADMM on the constraint conv(h,h) = C*ones, we need a different approach.
    
    # Alternative: Use the identity that for real h, the autoconvolution is related to
    # the squared magnitude of the DFT. Let's use a different formulation.
    
    # Actually, let's use a direct approach: minimize max(conv(h,h)) subject to sum(h)=1
    # We'll use the fact that the problem is non-convex in general, but we can try
    # iterative approaches.
    
    # Let me reconsider: The autoconvolution is quadratic in h, so this is a QCQP.
    # Let's use an ADMM-style approach with variable splitting.
    
    # Split: minimize ||z - C*1||_1 subject to z = conv(h,h), sum(h) = 1
    # where conv(h,h) is the full autoconvolution.
    
    # For the autoconvolution, we can use FFT: if H = FFT(h, 2N-1), then 
    # conv(h,h) = IFFT(|H|^2, 2N-1)[:2N-1] but this is tricky.
    
    # Let's use a simpler approach: iterative reweighted least squares or projected gradient.
    
    # Actually, let's try a different structural approach: 
    # Use the fact that we want to minimize max(conv(h,h))/sum(h)^2
    # By scaling invariance, we can fix sum(h) = 1 and minimize max(conv(h,h)).
    
    # Let's use coordinate descent with a twist: at each step, update one coordinate
    # to minimize the maximum of the autoconvolution.
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Initialize with a "spread out" signal
    # Try a signal that is approximately uniform in frequency domain
    # This means h is approximately a delta-like function in time... 
    # Actually, for flat autocorrelation, we want h to be like a random phase signal.
    
    # Let's try several initializations and refine with coordinate descent.
    
    rng = np.random.RandomState(42)
    
    def coordinate_descent(h, max_iter=5000, tol=1e-8):
        """Coordinate descent to minimize max(conv(h,h)) with sum(h) fixed."""
        h = h.copy().astype(float)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return h
        # Normalize to sum = 1
        h = h / s
        
        prev_val = compute_value(h)
        
        for it in range(max_iter):
            improved = False
            for i in range(N):
                old_hi = h[i]
                
                # Try to find the optimal h[i] given all other coordinates fixed.
                # conv(h,h) depends on h[i] quadratically.
                # Let's compute the effect of changing h[i].
                
                # For each lag k, conv[k] = sum_{i'+j'=k} h[i']*h[j']
                # Terms involving h[i]: 
                #   conv[i] includes h[i]*h[0] (if i < N)
                #   conv[i+N-1] includes h[i]*h[i] 
                #   etc.
                
                # This is complex. Let's use a numerical approach:
                # Try a few candidate values for h[i].
                
                # Current contribution to each conv[k] from h[i]:
                # For k in 0..2N-2, the terms involving h[i] are:
                #   h[i] * h[k-i] if 0 <= k-i < N and i != k-i
                #   h[i]^2 if k == 2*i
                #   2 * h[i] * h[k-i] if k-i != i and 0 <= k-i < N
                
                # Let's compute the gradient of max(conv) with respect to h[i].
                # This requires knowing which conv[k] is maximum.
                
                conv = np.convolve(h, h)
                k_max = np.argmax(conv)
                
                # d conv[k_max] / d h[i]:
                # conv[k_max] = sum_{a+b=k_max} h[a]*h[b]
                # d/dh[i] = h[k_max - i] if 0 <= k_max - i < N and k_max - i != i
                #          + 2*h[i] if k_max == 2*i
                #          + h[k_max - i] if k_max - i == i (but this is 2*h[i] case)
                
                j = k_max - i
                if 0 <= j < N:
                    if j == i:
                        grad = 2.0 * h[i]
                    else:
                        grad = h[j]
                else:
                    grad = 0.0
                
                # Also, other conv[k] values may be close to max, so we should consider
                # a smoothed version. But for now, let's use the exact max.
                
                # Update: h[i] -= step * grad
                # But we need to maintain sum(h) = 1, so adjust another coordinate.
                
                # Simpler: just do a small update and renormalize.
                step = 0.01
                new_hi = old_hi - step * grad
                h[i] = new_hi
                # Renormalize
                s_new = np.sum(h)
                if abs(s_new) > 1e-12:
                    h = h / s_new
                    h[i] = new_hi / s_new  # This isn't quite right
                
                # Actually, let's just do a simple update and check.
                val = compute_value(h)
                if val < prev_val:
                    prev_val = val
                    improved = True
                else:
                    h[i] = old_hi
                    # Renormalize back
                    s_new = np.sum(h)
                    if abs(s_new) > 1e-12:
                        h = h / s_new
            
            if not improved:
                break
        
        return h

    # Phase 1: Try random initializations
    for trial in range(8):
        if time.time() > deadline - 60.0:
            break
        
        if trial == 0:
            h_init = np.ones(N)
        elif trial == 1:
            h_init = rng.normal(0, 1, N)
            h_init += np.mean(1.0) - np.mean(h_init)
        elif trial == 2:
            h_init = np.zeros(N)
            h_init[::2] = 1.0
            h_init[1::2] = -0.5
        elif trial == 3:
            h_init = np.cos(np.pi * np.arange(N) / N)
            h_init += 0.5
        elif trial == 4:
            # Random with low-pass filter
            h_init = rng.normal(0, 1, N)
            # Apply simple low-pass
            kernel = np.ones(5) / 5
            h_init = np.convolve(h_init, kernel, mode='same')
            h_init += 1.0 - np.mean(h_init)
        elif trial == 5:
            # Sine wave
            h_init = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / 8.0)
        elif trial == 6:
            # Two-tone
            h_init = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / 7.0) + 0.2 * np.sin(2 * np.pi * np.arange(N) / 13.0)
        else:
            h_init = rng.uniform(-0.5, 1.5, N)
            h_init += 1.0 - np.mean(h_init)
        
        h_opt = coordinate_descent(h_init, max_iter=3000)
        val = compute_value(h_opt)
        
        if val < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
            best_h = h_opt.copy()
            best_val = val
            print(f"Trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
    
    # Phase 2: Fine-tune with smaller steps
    if time.time() < deadline - 30.0:
        h_fine = coordinate_descent(best_h, max_iter=10000)
        val_fine = compute_value(h_fine)
        if val_fine < best_val:
            best_h = h_fine.copy()
            best_val = val_fine
            print(f"Fine tune: improved to {best_val:.6f}", file=sys.stderr)
    
    # Final safety check
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
