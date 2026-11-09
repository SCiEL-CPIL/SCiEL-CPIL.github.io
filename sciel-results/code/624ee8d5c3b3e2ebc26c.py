# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak_sidelobe(h):
    """Compute max |R[k]| for k != 0 using FFT-based linear autocorrelation."""
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft - n + 1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))

def _greedy_construct(n):
    """
    Construct a binary sequence greedily to minimize the peak sidelobe of the prefix.
    """
    h = np.zeros(n)
    h[0] = 1.0
    
    for i in range(1, n):
        best_val = 1.0
        best_score = float('inf')
        
        for val in [1.0, -1.0]:
            h[i] = val
            prefix = h[:i+1]
            n_fft = 1 << (2 * (i + 1) - 1).bit_length()
            H = np.fft.rfft(prefix, n=n_fft)
            R = np.fft.irfft(H * np.conj(H), n=n_fft)
            
            sidelobes = np.concatenate([R[1:i+1], R[n_fft - i:n_fft]])
            score = np.max(np.abs(sidelobes))
            
            if score < best_score:
                best_score = score
                best_val = val
        
        h[i] = best_val
        
    return h

def _local_search(h, deadline, rng):
    """
    Local search using single flips, targeting the worst sidelobe.
    Includes random restarts/block flips if stuck.
    """
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    best_h = h.copy()
    best_score = _autocorr_peak_sidelobe(h)
    h = h.copy()
    
    stuck_count = 0
    max_stuck = 100
    
    while time.time() < deadline:
        # Find the worst sidelobe
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        
        # Get all sidelobes
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        
        max_val = np.max(np.abs(all_sidelobes))
        if max_val == 0:
            break
            
        idx = np.argmax(np.abs(all_sidelobes))
        
        # Determine lag magnitude k
        if idx < n - 1:
            k = idx + 1
        else:
            r = idx - (n - 1)
            k = n - 1 - r
            
        # k is the absolute lag
        r_val = R[k]
        s = np.sign(r_val)
        if s == 0:
            s = 1.0
            
        # Vectorized calculation of neighbor_sum for lag k
        neighbor_sum = np.zeros(n)
        if k > 0:
            neighbor_sum[:n-k] += h[k:]
            neighbor_sum[k:] += h[:n-k]
        
        # Calculate potential reduction for each j
        # reduction[j] = 2 * s * h[j] * neighbor_sum[j]
        reduction = 2.0 * s * h * neighbor_sum
        
        best_j = np.argmax(reduction)
        improved = False
        
        if reduction[best_j] > 0:
            h_new = h.copy()
            h_new[best_j] *= -1
            new_score = _autocorr_peak_sidelobe(h_new)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h_new
                h = h_new
                improved = True
                stuck_count = 0
            else:
                stuck_count += 1
        else:
            stuck_count += 1
            
        # If stuck, try a random block flip
        if not improved and stuck_count > max_stuck:
            # Random block flip
            block_size = rng.integers(2, 6)
            start = rng.integers(0, n - block_size + 1)
            h_new = h.copy()
            h_new[start:start+block_size] *= -1
            new_score = _autocorr_peak_sidelobe(h_new)
            
            if new_score < best_score:
                best_score = new_score
                best_h = h_new
                h = h_new
                stuck_count = 0
            else:
                # Accept with small probability to escape?
                # Or just reset stuck count to try again
                stuck_count = 0
                
    return best_h

def construct():
    t_start = time.time()
    deadline = t_start + 130.0
    
    # Phase 1: Greedy Construction
    h = _greedy_construct(N)
    
    # Phase 2: Local Search
    rng = np.random.default_rng(42)
    h = _local_search(h, deadline, rng)
    
    return h
# EVOLVE-BLOCK-END
