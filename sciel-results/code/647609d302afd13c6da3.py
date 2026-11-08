import numpy as np
import time
import random

N = 100

def compute_overlap_fast(h):
    """Compute overlap at all shifts using FFT. h is array of length N."""
    # overlap(k) = sum_i h[i] * (1-h[(i+k) mod N])
    #            = N/2 - sum_i h[i] * h[(i+k) mod N]
    # The second term is the circular autocorrelation of h
    n = len(h)
    # Use FFT to compute circular autocorrelation
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i] * h[(i+k) mod N]
    # overlap(k) = n/2 - autocorr[k]
    overlaps = n / 2.0 - autocorr
    return overlaps

def score(h):
    """Return the max overlap."""
    return np.max(compute_overlap_fast(h))

def construct():
    start = time.time()
    deadline = 140.0  # safety margin
    
    # Start with uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = score(h)
    
    # Try binary patterns: sequences of 0s and 1s with exactly N/2 ones
    # This is related to finding sequences with good autocorrelation properties
    
    random.seed(42)
    np.random.seed(42)
    
    # Try several binary patterns
    def try_binary_pattern(pattern):
        """Try a binary pattern, return score."""
        h_bin = np.array(pattern, dtype=float)
        if len(h_bin) != N or np.sum(h_bin) != N/2:
            return None
        s = score(h_bin)
        return s
    
    # Try some known good patterns
    # Pattern: alternating 0,1,0,1,... (but this gives N/2 ones only if N is even)
    # For N=100, alternating gives 50 ones
    alt_pattern = [i % 2 for i in range(N)]
    s_alt = try_binary_pattern(alt_pattern)
    if s_alt is not None and s_alt < best_score:
        best_score = s_alt
        best_h = np.array(alt_pattern, dtype=float)
    
    # Try m-sequence-like patterns or other structured patterns
    # Try all shifts of a base pattern
    # Try random binary patterns with exactly N/2 ones
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Create random binary pattern with exactly N/2 ones
        indices = list(range(N))
        random.shuffle(indices)
        ones_positions = indices[:N//2]
        h_bin = np.zeros(N)
        for pos in ones_positions:
            h_bin[pos] = 1.0
        
        s = score(h_bin)
        if s < best_score:
            best_score = s
            best_h = h_bin.copy()
    
    # Now try to improve by allowing non-binary values
    # Use local search from the best solution
    current_h = best_h.copy()
    current_score = best_score
    
    # Local search: try small perturbations
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random index and perturb it
        idx = random.randint(0, N-1)
        delta = random.uniform(-0.05, 0.05)
        new_h = current_h.copy()
        new_h[idx] = np.clip(new_h[idx] + delta, 0, 1)
        
        # Adjust another index to maintain sum constraint
        diff = N/2.0 - np.sum(new_h)
        if abs(diff) > 1e-10:
            # Find an index that can absorb the adjustment
            candidates = [j for j in range(N) if j != idx]
            random.shuffle(candidates)
            adjusted = False
            for other_idx in candidates:
                adj = new_h[other_idx] - diff
                if 0 <= adj <= 1:
                    new_h[other_idx] = adj
                    adjusted = True
                    break
            
            if not adjusted:
                continue
        
        # Verify sum is correct
        if abs(np.sum(new_h) - N/2.0) > 1e-8:
            continue
        
        s = score(new_h)
        if s < current_score - 1e-10:
            current_h = new_h
            current_score = s
            if s < best_score:
                best_score = s
                best_h = new_h.copy()
    
    return best_h
