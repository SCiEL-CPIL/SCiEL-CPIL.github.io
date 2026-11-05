"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit
    
    # Start with a Zadoff-Chu inspired real chirp sequence
    # z[k] = exp(i*pi*k^2/L), use real part
    # For length L=2N, the autoconvolution is flat
    L = 2 * N
    k = np.arange(N)
    h = np.cos(np.pi * k * k / L)
    
    # Ensure sum is not zero
    s = np.sum(h)
    if abs(s) < 1e-10:
        h = h + 1e-6
        s = np.sum(h)
    
    # Compute current value
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-15:
            return float('inf')
        conv = np.convolve(h, h)
        mx = np.max(conv)
        return abs(2 * N * mx / (s * s))
    
    best_h = h.copy()
    best_val = compute_value(h)
    
    # Try to improve with iterative refinement
    # The objective is to make the autoconvolution as flat as possible
    # Use gradient descent on the smoothed objective
    
    # Try several different chirp parameters
    candidates = [h.copy()]
    
    # Different chirp rates
    for alpha in [0.5, 1.0, 1.5, 2.0, 0.25, 0.75]:
        k2 = np.arange(N)
        h_cand = np.cos(alpha * np.pi * k2 * k2 / L)
        s_cand = np.sum(h_cand)
        if abs(s_cand) > 1e-10:
            candidates.append(h_cand)
    
    # Also try sine
    for alpha in [0.5, 1.0, 1.5]:
        k2 = np.arange(N)
        h_cand = np.sin(alpha * np.pi * k2 * k2 / L)
        s_cand = np.sum(h_cand)
        if abs(s_cand) > 1e-10:
            candidates.append(h_cand)
    
    # Evaluate all candidates
    for hc in candidates:
        v = compute_value(hc)
        if v < best_val:
            best_val = v
            best_h = hc.copy()
    
    # Now do gradient descent to further optimize
    # Objective: f(h) = 2*N*max(conv(h,h)) / sum(h)^2
    # We want to minimize this (assuming it's positive)
    # Use a soft-max surrogate: max(x) ≈ log(sum(exp(x/tau))) * tau for large tau... 
    # Actually, let's use a different approach: minimize the max of conv entries
    # subject to sum(h) = 1 (normalize)
    
    # Normalize to sum = 1
    h_norm = best_h / np.sum(best_h)
    
    # Gradient descent on normalized h
    # conv(h,h) is quadratic in h, so we can compute gradients
    # d/dh_j [conv(h,h)[m]] = h[m-j] + h[m-j+N] (with appropriate bounds)
    
    # Actually, let's use a simpler approach: coordinate descent
    # For each coordinate, find the optimal value given others
    
    # Let's try a few iterations of projected gradient descent
    # Minimize: max_m conv(h,h)[m] / (sum(h))^2
    # With constraint sum(h) = 1
    
    lr = 0.01
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        h_cur = h_norm.copy()
        s = np.sum(h_cur)
        if abs(s) < 1e-15:
            break
        
        conv = np.convolve(h_cur, h_cur)
        # Find the max entry
        max_idx = np.argmax(conv)
        max_val = conv[max_idx]
        
        # Gradient of conv[h,m] with respect to h[j]:
        # conv[h,m] = sum_j h[j]*h[m-j]
        # d/dh[j] = h[m-j] + (if m-j+N in range) h[m-j+N]... 
        # Actually for full convolution: conv[h,m] = sum_{j=max(0,m-N+1)}^{min(N-1,m)} h[j]*h[m-j]
        # d/dh[j] conv[h,m] = h[m-j] if m-j in [0,N-1]
        # But also from the other term: when we write conv = h * h, 
        # conv[m] = sum_j h[j] h[m-j]
        # d/dh[j] = h[m-j] + h[m-j] if we consider both... no
        # Actually conv[m] = sum_j h[j] h[m-j], so d/dh[j] = h[m-j] (for valid m-j)
        # But wait, h[m-j] is also a function of h, so:
        # d/dh[j] [sum_k h[k] h[m-k]] = h[m-j] + h[m-j] = 2*h[m-j] if m-j is in range
        # No wait: d/dh[j] of h[k]*h[m-k] is nonzero when k=j or m-k=j
        # When k=j: derivative is h[m-j]
        # When m-k=j i.e. k=m-j: derivative is h[j] (if m-j != j)
        # So d/dh[j] = h[m-j] + h[m-j] = 2*h[m-j] if m-j in [0,N-1] and m-j != j
        # If m-j == j, then d/dh[j] = 2*h[j]
        
        # Simpler: use numerical gradient or just use the fact that
        # the gradient of conv[m] w.r.t. h is a vector where entry j is:
        # h[m-j] if 0 <= m-j < N, else 0, PLUS h[m-j] if m-j is valid from the other side
        # Actually for autoconvolution: grad_j = h[m-j] + h[m-j] ... no
        # Let me just compute it numerically for the max entry
        
        # Actually, let's use a different approach: 
        # minimize sum of squares of (conv[m] - mean(conv)) to flatten
        # This is smoother and easier to optimize
        
        # Or even simpler: use the fact that for Zadoff-Chu, the sequence is optimal
        # Let's just try to fine-tune the chirp parameter
        
        break  # Skip complex optimization for now, try parameter sweep
    
    # Try a fine sweep of the chirp parameter
    best_val_sweep = best_val
    best_h_sweep = best_h.copy()
    
    k_arr = np.arange(N)
    for i in range(50):
        if time.time() > deadline - 5.0:
            break
        alpha = 0.8 + 0.04 * i  # sweep around 1.0
        h_cand = np.cos(alpha * np.pi * k_arr * k_arr / L)
        s_cand = np.sum(h_cand)
        if abs(s_cand) < 1e-10:
            continue
        v = compute_value(h_cand)
        if v < best_val_sweep:
            best_val_sweep = v
            best_h_sweep = h_cand.copy()
    
    if best_val_sweep < best_val:
        best_h = best_h_sweep
        best_val = best_val_sweep
    
    # Also try negative chirps and other phases
    for phase in [0.0, 0.5, 1.0]:
        if time.time() > deadline - 5.0:
            break
        for alpha in [0.9, 0.95, 1.0, 1.05, 1.1]:
            h_cand = np.cos(alpha * np.pi * k_arr * k_arr / L + phase * np.pi)
            s_cand = np.sum(h_cand)
            if abs(s_cand) < 1e-10:
                continue
            v = compute_value(h_cand)
            if v < best_val:
                best_val = v
                best_h = h_cand.copy()
    
    # Final check
    final_val = compute_value(best_h)
    if final_val != best_val:
        best_val = final_val
    
    return best_h
# EVOLVE-BLOCK-END
