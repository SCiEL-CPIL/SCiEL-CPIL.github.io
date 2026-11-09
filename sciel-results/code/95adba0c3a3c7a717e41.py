"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_overlap_profile(h):
    """
    Compute the overlap of h with shifted (1-h) for all cyclic shifts.
    
    h is a sequence of length N with sum N/2.
    1-h has sum N/2.
    
    For a shift k, the overlap is sum_i h[i] * (1-h)[(i-k) % N]
    = sum_i h[i] * (1 - h[(i-k) % N])
    = sum_i h[i] - sum_i h[i]*h[(i-k) % N]
    = N/2 - autocorr_h[k]
    
    where autocorr_h[k] = sum_i h[i] * h[(i-k) % N]
    
    We want to minimize the maximum overlap over all shifts k.
    max_k overlap(k) = N/2 - min_k autocorr_h[k]
    
    So minimizing max overlap is equivalent to minimizing (N/2 - min autocorr),
    which is equivalent to maximizing min_k autocorr_h[k].
    """
    # Use FFT to compute circular autocorrelation
    H = np.fft.rfft(h, n=N)
    autocorr = np.fft.irfft(H * np.conj(H), n=N)[:N]
    # autocorr[k] = sum_i h[i] * h[(i-k) % N] for k in 0..N-1
    # Actually, let's be careful with the indexing.
    # np.fft.irfft of |H|^2 gives the circular autocorrelation.
    # autocorr[k] = sum_i h[i] * h[(i+k) % N] typically, or similar.
    # Let's just compute it directly for clarity and correctness, 
    # but FFT is faster. Let's verify the sign/indexing.
    # Standard circular autocorrelation: R[k] = sum_i x[i] x[(i-k) mod N]
    # Using FFT: R = ifft(fft(x) * conj(fft(x)))
    # This gives R[k] = sum_i x[i] x[(i-k) mod N]
    
    # The overlap for shift k is: sum_i h[i] * (1-h)[(i-k)%N]
    # (1-h)[j] = 1 - h[j]
    # So overlap[k] = sum_i h[i] * (1 - h[(i-k)%N])
    #                = sum_i h[i] - sum_i h[i] h[(i-k)%N]
    #                = N/2 - R[k]
    
    # We want to minimize max_k overlap[k] = N/2 - min_k R[k]
    # So we want to maximize min_k R[k].
    
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap


def _local_search(h, max_iter=5000, time_limit=100.0):
    """
    Local search on the step heights.
    We perturb individual elements or small groups, maintaining sum = N/2 and bounds [0,1].
    """
    h = h.copy()
    current_score = _compute_overlap_profile(h)
    best_h = h.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Simple local search: try adjusting pairs of elements to maintain sum
    # Or adjust one element and compensate in another.
    # Let's try a more structured approach: random restarts + local refinement.
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random indices to swap/adjust
        i, j = np.random.choice(N, 2, replace=False)
        
        # Try to increase h[i] by delta and decrease h[j] by delta
        # delta must be such that h[i]+delta <= 1 and h[j]-delta >= 0
        delta_options = []
        # Try a few delta values
        base_delta = 0.1
        for mult in [0.5, 1.0, 2.0]:
            delta = base_delta * mult
            if h[i] + delta <= 1.0 and h[j] - delta >= 0.0:
                delta_options.append(delta)
            if h[i] - delta >= 0.0 and h[j] + delta <= 1.0:
                delta_options.append(-delta)
        
        if not delta_options:
            continue
            
        best_delta = None
        best_new_score = current_score
        
        for delta in delta_options:
            h_new = h.copy()
            h_new[i] += delta
            h_new[j] -= delta
            
            # Check bounds
            if np.any(h_new < -1e-9) or np.any(h_new > 1.0 + 1e-9):
                continue
                
            new_score = _compute_overlap_profile(h_new)
            if new_score < best_new_score:
                best_new_score = new_score
                best_delta = delta
        
        if best_delta is not None:
            h[i] += best_delta
            h[j] -= best_delta
            h = np.clip(h, 0, 1)
            # Renormalize to ensure sum is exactly N/2 (floating point)
            h = h - (np.sum(h) - N/2.0) / N
            h = np.clip(h, 0, 1)
            # This renormalization might break bounds slightly, re-clip
            # A better way is to adjust the last element, but that's complex.
            # For now, the clip is fine if we're careful.
            
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
    
    return best_h, best_score


def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Initialize with constant 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _compute_overlap_profile(h)
    
    # Try a few random restarts with local search
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        # Generate random values in [0,1] and normalize to sum N/2
        h_rand = np.random.rand(N)
        h_rand = h_rand / np.sum(h_rand) * (N / 2.0)
        h_rand = np.clip(h_rand, 0, 1)
        # Adjust to ensure sum is exactly N/2
        diff = N / 2.0 - np.sum(h_rand)
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0, 1)
        
        # Local search
        h_ls, score_ls = _local_search(h_rand, max_iter=2000, time_limit=time_limit - (time.time() - start_time))
        
        if score_ls < best_score:
            best_score = score_ls
            best_h = h_ls
    
    # Final check
    # Ensure the result is valid
    best_h = np.clip(best_h, 0, 1)
    # Adjust sum if necessary
    diff = N / 2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
