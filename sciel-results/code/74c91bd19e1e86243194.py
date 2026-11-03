"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Use fixed-point iteration with soft-min regularization to find h that minimizes
    the maximum overlap of h with shifted 1-h.
    
    The approach:
    1. O(s) = N/2 - autocorrelation(s) where autocorrelation(s) = sum_i h[i]*h[(i+s)%N]
    2. We want to minimize max_s O(s), equivalent to maximizing min_s autocorrelation(s)
    3. Use fixed-point iteration: h_new = project(h - eta * gradient)
       where gradient comes from the soft-min of overlaps
    
    The soft-min approximation: min_s f(s) ≈ -ln(sum exp(-f(s)/t)) * t for small t
    This gives a smooth objective that we can use in gradient descent.
    
    We iterate:
    1. Compute autocorrelations ac(s) for all shifts s
    2. Compute overlaps O(s) = N/2 - ac(s)
    3. Use soft-min to get a smooth approximation of max_s O(s)
    4. Compute gradient and update h
    5. Project onto feasible set (sum = N/2, 0 <= h <= 1)
    """
    N_local = N
    target_sum = N_local / 2.0
    
    # Initialize with uniform
    h = np.full(N_local, 0.5)
    
    # Parameters for soft-min and iteration
    temperature = 0.1  # Soft-min temperature (smaller = sharper)
    lr = 0.1  # Learning rate
    n_iter = 200  # Number of iterations
    
    # Precompute shift indices
    shifts = np.arange(N_local)
    
    for it in range(n_iter):
        # Compute autocorrelations for all shifts using FFT
        # autocorrelation(s) = sum_i h[i] * h[(i+s) % N]
        # This is circular autocorrelation
        h_fft = np.fft.fft(h)
        ac_fft = h_fft * np.conj(h_fft)
        ac = np.real(np.fft.ifft(ac_fft))
        
        # Overlaps: O(s) = N/2 - ac(s)
        overlaps = target_sum - ac
        
        # Soft-max of overlaps (approximate max)
        # max_s O(s) ≈ t * ln(sum exp(O(s)/t))
        # Gradient with respect to h: d/dh [t * ln(sum exp(O(s)/t))]
        # = t * (1/sum exp(O(s)/t)) * sum exp(O(s)/t) * (1/t) * dO(s)/dh
        # = (sum exp(O(s)/t) * dO(s)/dh) / sum exp(O(s)/t)
        # = sum w(s) * dO(s)/dh where w(s) = exp(O(s)/t) / sum exp(O(s)/t)
        
        # Normalize for numerical stability
        max_overlap = np.max(overlaps)
        exp_overlaps = np.exp((overlaps - max_overlap) / temperature)
        weights = exp_overlaps / np.sum(exp_overlaps)
        
        # Compute gradient of weighted sum of overlaps
        # O(s) = N/2 - sum_i h[i]*h[(i+s)%N]
        # dO(s)/dh[j] = -sum_i (delta_{i,j} * h[(i+s)%N] + h[i] * delta_{(i+s)%N, j})
        # = -h[(j+s)%N] - h[(j-s)%N]
        # (using circular indexing)
        
        grad = np.zeros(N_local)
        for s in range(N_local):
            w = weights[s]
            if w > 1e-10:
                # dO(s)/dh[j] = -h[(j+s)%N] - h[(j-s)%N]
                grad -= w * (np.roll(h, s) + np.roll(h, -s))
        
        # Update h
        h_new = h - lr * grad
        
        # Project onto feasible set:
        # 1. Clip to [0, 1]
        h_new = np.clip(h_new, 0, 1)
        
        # 2. Adjust to meet sum constraint
        current_sum = np.sum(h_new)
        if abs(current_sum - target_sum) > 1e-10:
            diff = target_sum - current_sum
            # Distribute difference, respecting bounds
            if diff > 0:
                # Need to add to elements not at 1
                idx = np.where(h_new < 1)[0]
                if len(idx) > 0:
                    add_per = min(diff / len(idx), 1 - h_new[idx])
                    # Simple approach: add evenly then clip
                    h_new[idx] += diff / len(idx)
                    h_new = np.clip(h_new, 0, 1)
            else:
                # Need to subtract from elements not at 0
                idx = np.where(h_new > 0)[0]
                if len(idx) > 0:
                    h_new[idx] += diff / len(idx)
                    h_new = np.clip(h_new, 0, 1)
            
            # Final adjustment if still off
            for _ in range(50):
                current_sum = np.sum(h_new)
                if abs(current_sum - target_sum) < 1e-12:
                    break
                if current_sum < target_sum:
                    idx = np.where(h_new < 1)[0]
                    if len(idx) == 0:
                        break
                    h_new[idx] += (target_sum - current_sum) / len(idx)
                else:
                    idx = np.where(h_new > 0)[0]
                    if len(idx) == 0:
                        break
                    h_new[idx] -= (current_sum - target_sum) / len(idx)
                h_new = np.clip(h_new, 0, 1)
        
        h = h_new
    
    # Final exact sum adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        for _ in range(100):
            current_sum = np.sum(h)
            if abs(current_sum - target_sum) < 1e-12:
                break
            if current_sum < target_sum:
                idx = np.where(h < 1)[0]
                if len(idx) == 0:
                    break
                h[idx] += (target_sum - current_sum) / len(idx)
            else:
                idx = np.where(h > 0)[0]
                if len(idx) == 0:
                    break
                h[idx] -= (current_sum - target_sum) / len(idx)
            h = np.clip(h, 0, 1)
    
    return h
# EVOLVE-BLOCK-END
