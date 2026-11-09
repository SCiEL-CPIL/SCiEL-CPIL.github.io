"""
Program for the third autocorrelation inequality task.
Strategy: Random initialization + Greedy Local Search (Hill Climbing).
This approach abandons the structured LFSR sequence in favor of a random
starting point, which typically has lower peak sidelobe autocorrelation
for aperiodic metrics, and then refines the sequence by flipping bits
that reduce the maximum absolute autocorrelation value.
"""
import numpy as np
import time

N = 400

def _peak_sidelobe(signs):
    """
    Compute the peak absolute aperiodic autocorrelation for non-zero lags.
    Returns (peak_value, worst_lag_index).
    """
    # Efficient autocorrelation via FFT
    # Length 2*N is sufficient for aperiodic autocorrelation of length N
    f = np.fft.fft(signs, 2 * N)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft(power).real
    
    # The aperiodic autocorrelation R[k] for k=0..N-1 is in autocorr[0:N]
    # R[k] = sum_{i=0}^{N-1-k} s[i] * s[i+k]
    # We want max |R[k]| for k in 1..N-1
    sidelobes = np.abs(autocorr[1:N])
    peak = np.max(sidelobes)
    worst_lag = np.argmax(sidelobes) + 1 # Lag index is 1-based in the sidelobes array
    
    return peak, worst_lag

def _flip_bit_effect(signs, idx, worst_lag):
    """
    Estimate the change in the autocorrelation at the worst_lag if we flip signs[idx].
    We only need to evaluate the change at the current worst lag to decide if a flip helps.
    However, flipping a bit affects all lags. To be safe and effective, we check the new peak
    after the flip in the main loop, but we can use this heuristic to filter candidates.
    
    Actually, a simpler heuristic: 
    R[k] = sum s[i] s[i+k].
    If we flip s[idx], terms involving s[idx] change sign.
    R_new[k] = R[k] - 2 * s[idx] * (sum_{j: idx in {j, j+k}} s[j])
    
    Let's just try flipping bits that are "aligned" with the worst lag's contribution.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Random Initialization
    # Use a fixed seed for reproducibility, or random. Random is better for diversity.
    # We'll try a few random starts if time permits, but one good one is often enough.
    np.random.seed(42)
    signs = np.random.choice([-1, 1], size=N)
    
    current_peak, current_worst_lag = _peak_sidelobe(signs)
    best_peak = current_peak
    best_signs = signs.copy()
    
    # 2. Greedy Local Search
    # We iterate through the sequence and flip bits if it reduces the peak sidelobe.
    # To speed up, we can be selective about which bits to try flipping.
    # A bit at index i affects lags k where the term s[i]*s[i+k] is part of the sum.
    # Specifically, flipping s[i] changes R[k] for k such that i < N-k (forward) or i+k < N (backward).
    # Actually, R[k] = sum_{j=0}^{N-1-k} s[j]s[j+k].
    # s[i] appears in R[k] if:
    #   i = j  => s[i]s[i+k] (requires i+k < N)
    #   i = j+k => s[i-k]s[i] (requires i-k >= 0)
    
    # We will do multiple passes. In each pass, we scan indices 0..N-1.
    # If flipping index i reduces the global peak, we keep the flip.
    
    max_iterations = 50  # Number of full passes
    pass_count = 0
    
    while time.time() < deadline and pass_count < max_iterations:
        pass_count += 1
        flipped_any = False
        
        # Shuffle the order of indices to avoid bias towards the start of the array
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try flipping bit i
            old_val = signs[i]
            signs[i] = -old_val
            
            new_peak, new_worst_lag = _peak_sidelobe(signs)
            
            if new_peak < current_peak:
                # Improvement found
                current_peak = new_peak
                current_worst_lag = new_worst_lag
                flipped_any = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_val
                
        # If no flips were made in a full pass, we are at a local minimum.
        if not flipped_any:
            break
            
    # 3. Return the best sequence found
    # The task expects a sequence of signed step heights.
    # The interface says "sequence of signed step heights".
    # The seed returned np.ones(N) which are +1.
    # So we return the signs array (+1/-1).
    
    return best_signs

# Ensure the function is accessible
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result shape: {res.shape}")
    print(f"First few: {res[:10]}")
    p, _ = _peak_sidelobe(res)
    print(f"Peak Sidelobe: {p}")
    print(f"Normalized (p/N): {p/N}")
