"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_overlap(h):
    """Compute max over shifts k of sum_i h[i]*(1-h[(i+k)%N])"""
    h = np.asarray(h)
    n = len(h)
    max_ov = 0.0
    for k in range(1, n):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective(h_flat):
    h = np.asarray(h_flat)
    # Penalize constraint violations
    sum_h = np.sum(h)
    target = N / 2.0
    # Soft penalty for sum constraint
    penalty_sum = 100.0 * (sum_h - target) ** 2
    # Penalty for out-of-bounds (shouldn't happen with bounds, but safety)
    penalty_bounds = 100.0 * np.sum(np.maximum(0, h - 1.0) ** 2) + 100.0 * np.sum(np.maximum(0, -h) ** 2)
    ov = _compute_max_overlap(h)
    return ov + penalty_sum + penalty_bounds

def _compute_overlap_fast(h):
    """Faster overlap computation using FFT for all shifts at once."""
    n = len(h)
    # overlap(k) = sum_i h[i]*(1-h[(i+k)%n]) = sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    # = n/2 - autocorr(k)
    # Use FFT to compute circular autocorrelation
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i]*h[(i-k)%n] (circular)
    # We want sum_i h[i]*h[(i+k)%n] which is autocorr[-k % n] = autocorr[n-k]
    # overlap(k) = n/2 - autocorr[(n-k)%n] for k=1..n-1
    overlaps = n / 2.0 - np.concatenate([autocorr[n-1:0:-1], autocorr[1:]])
    # Actually let me be more careful
    # autocorr[m] = sum_i h[i] * h[(i-m) % n]
    # We want C(k) = sum_i h[i] * h[(i+k) % n] = autocorr[(-k) % n] = autocorr[(n-k)%n]
    # For k=1: C(1) = autocorr[n-1]
    # For k=2: C(2) = autocorr[n-2]
    # ...
    # For k=n-1: C(n-1) = autocorr[1]
    C = np.concatenate([autocorr[n-1:0:-1], autocorr[1:]])  # C[k-1] = C(k) for k=1..n-1
    overlaps = n / 2.0 - C
    return np.max(overlaps)

def _objective_fast(h_flat):
    h = np.asarray(h_flat)
    sum_h = np.sum(h)
    target = N / 2.0
    penalty_sum = 50.0 * (sum_h - target) ** 2
    penalty_bounds = 50.0 * (np.sum(np.maximum(0, h - 1.0) ** 2) + np.sum(np.maximum(0, -h) ** 2))
    ov = _compute_overlap_fast(h)
    return ov + penalty_sum + penalty_bounds

def _objective_grad(h_flat):
    """Objective with gradient using FFT."""
    h = np.asarray(h_flat)
    n = len(h)
    sum_h = np.sum(h)
    target = n / 2.0
    
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    C = np.concatenate([autocorr[n-1:0:-1], autocorr[1:]])
    overlaps = n / 2.0 - C
    max_ov = np.max(overlaps)
    
    # Gradient of max overlap w.r.t. h
    # d(overlap_k)/dh[i] = (1 - h[(i+k)%n]) - h[i]*(1 - h[(i-k)%n])... 
    # This is complex. Let's just use numerical gradient or L-BFGS-B with finite diffs.
    
    penalty_sum = 50.0 * (sum_h - target) ** 2
    penalty_bounds = 50.0 * (np.sum(np.maximum(0, h - 1.0) ** 2) + np.sum(np.maximum(0, -h) ** 2))
    
    return max_ov + penalty_sum + penalty_bounds

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    bounds = [(0.0, 1.0)] * N
    target_sum = N / 2.0
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try multiple initializations with different Fourier modes
    rng = np.random.RandomState(42)
    
    num_starts = 20
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Create diverse initializations
        if start_idx == 0:
            h0 = np.full(N, 0.5)
        elif start_idx == 1:
            # Binary-like: alternating
            h0 = np.zeros(N)
            h0[::2] = 1.0
        elif start_idx == 2:
            # Binary-like: blocks
            h0 = np.zeros(N)
            h0[:N//2] = 1.0
        else:
            # Random Fourier mode perturbations
            h0 = np.full(N, 0.5)
            # Add random low-frequency components
            num_modes = rng.randint(1, 5)
            for _ in range(num_modes):
                freq = rng.randint(1, N // 2 + 1)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.1, 0.4)
                h0 += amp * np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
            h0 = np.clip(h0, 0.0, 1.0)
            # Rescale to have correct sum
            if np.sum(h0) > 0:
                h0 = h0 * (target_sum / np.sum(h0))
                h0 = np.clip(h0, 0.0, 1.0)
        
        # Optimize
        try:
            result = minimize(
                _objective_fast,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            
            h_opt = result.x
            # Project to valid region
            h_opt = np.clip(h_opt, 0.0, 1.0)
            # Fix sum
            sum_h = np.sum(h_opt)
            if sum_h > 0:
                h_opt = h_opt * (target_sum / sum_h)
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            # Verify sum is close to target
            if abs(np.sum(h_opt) - target_sum) > 0.01:
                # Simple correction
                diff = target_sum - np.sum(h_opt)
                h_opt = h_opt + diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _compute_max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Final polish: local optimization with tighter settings
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective_fast,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            h_opt = np.clip(result.x, 0.0, 1.0)
            sum_h = np.sum(h_opt)
            if sum_h > 0:
                h_opt = h_opt * (target_sum / sum_h)
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _compute_max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    sum_h = np.sum(best_h)
    if abs(sum_h - target_sum) > 1e-10:
        diff = target_sum - sum_h
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-fix if clipping changed sum
        sum_h2 = np.sum(best_h)
        if abs(sum_h2 - target_sum) > 1e-10:
            diff2 = target_sum - sum_h2
            best_h = best_h + diff2 / N
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
