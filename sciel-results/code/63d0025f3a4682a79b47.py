"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h over all cyclic shifts."""
    # h is an array of length N, values in [0,1]
    # We want to minimize max over shifts k of sum_i h[i] * (1 - h[(i+k) % N])
    # This is equivalent to: for each shift k, compute sum_i h[i] - h[i]*h[(i+k)%N]
    # = sum(h) - sum_i h[i]*h[(i+k)%N]
    # Since sum(h) = N/2 is fixed, minimizing max overlap is equivalent to
    # maximizing min over k of sum_i h[i]*h[(i+k)%N]
    
    # Actually let's re-read: "max overlap of h with shifted 1-h"
    # Overlap of h with shifted (1-h) at shift k = sum_i h[i] * (1-h[(i+k)%N])
    # = sum(h) - sum_i h[i]*h[(i+k)%N]
    # = N/2 - autocorrelation at lag k
    
    # So we want to minimize max_k [N/2 - autocorr(k)] = N/2 - min_k autocorr(k)
    # To minimize this, we want to maximize min_k autocorr(k)
    
    # Let's compute all autocorrelations
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    
    # Use FFT for efficiency
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    # Circular autocorrelation via FFT
    H = np.fft.fft(h_pad, n=2*N)
    autocorr_full = np.fft.ifft(H * np.conj(H)).real[:N]
    
    # autocorr_full[k] should give sum_i h[i]*h[(i+k)%N] for k=0..N-1
    # But let's verify: for circular autocorrelation of length N
    # Actually with zero-padding to 2N and taking first N, we get linear autocorrelation
    # Let me just compute it directly for correctness, N=100 is small
    
    # Direct computation
    autocorr = np.zeros(N)
    for k in range(N):
        autocorr[k] = np.sum(h * np.roll(h, -k))
    
    # Overlap at shift k = N/2 - autocorr[k]
    # Max overlap = N/2 - min(autocorr)
    # We want to minimize this, i.e., maximize min(autocorr)
    
    max_overlap = N / 2.0 - np.min(autocorr)
    return max_overlap

def _objective(params, N):
    """Objective function for optimization.
    
    params: knot values (length M), will be interpolated to N points
    Returns the maximum overlap value.
    """
    M = len(params)
    # Create knot positions
    knot_pos = np.linspace(0, N-1, M)
    # Interpolate to get N values
    h = np.interp(np.arange(N), knot_pos, params, period=N)
    # Clip to [0, 1]
    h = np.clip(h, 0, 1)
    # Normalize so that sum = N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    else:
        h = np.full(N, 0.5)
    # Re-clip after scaling (scaling might push above 1)
    h = np.clip(h, 0, 1)
    # Re-normalize
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    # Final clip and normalize
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    
    return _overlap_max(h)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, use 115 for safety
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try different numbers of knots
    for M in [4, 6, 8, 10, 12, 16, 20]:
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with alternating pattern
        init = np.zeros(M)
        for i in range(M):
            if i % 2 == 0:
                init[i] = 1.0
            else:
                init[i] = 0.0
        
        # Also try a sinusoidal initialization
        init2 = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(M) / M * 2)
        
        for init_params in [init, init2]:
            if time.time() > deadline - 5.0:
                break
            
            try:
                result = minimize(
                    _objective,
                    init_params,
                    args=(N,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                if result.fun < best_score:
                    # Convert back to h
                    M_used = len(result.x)
                    knot_pos = np.linspace(0, N-1, M_used)
                    h_new = np.interp(np.arange(N), knot_pos, result.x, period=N)
                    h_new = np.clip(h_new, 0, 1)
                    s = np.sum(h_new)
                    if s > 0:
                        h_new = h_new * (N / 2.0) / s
                    h_new = np.clip(h_new, 0, 1)
                    s = np.sum(h_new)
                    if s > 0:
                        h_new = h_new * (N / 2.0) / s
                    h_new = np.clip(h_new, 0, 1)
                    
                    actual_score = _overlap_max(h_new)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_h = h_new
                        
            except Exception:
                pass
    
    # Final polish: try direct optimization on all N values with a few iterations
    if time.time() < deadline - 10.0:
        try:
            # Use the best_h as starting point
            h_current = best_h.copy()
            
            # Do a few rounds of coordinate descent
            for iteration in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                improved = False
                for i in range(N):
                    old_val = h_current[i]
                    # Try small perturbations
                    for delta in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                        h_test = h_current.copy()
                        h_test[i] = np.clip(old_val + delta, 0, 1)
                        # Renormalize
                        s = np.sum(h_test)
                        if s > 0:
                            h_test = h_test * (N / 2.0) / s
                        h_test = np.clip(h_test, 0, 1)
                        s = np.sum(h_test)
                        if s > 0:
                            h_test = h_test * (N / 2.0) / s
                        h_test = np.clip(h_test, 0, 1)
                        
                        score = _overlap_max(h_test)
                        if score < best_score - 1e-10:
                            best_score = score
                            best_h = h_test.copy()
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
                    
        except Exception:
            pass
    
    # Ensure exact sum = N/2
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
    best_h = np.clip(best_h, 0, 1)
    
    # Final adjustment to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute the difference
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        # If clipping changed things, try again
        s = np.sum(best_h)
        if abs(s - N/2.0) > 1e-10:
            best_h = best_h * (N / 2.0) / s
    
    return best_h
# EVOLVE-BLOCK-END
