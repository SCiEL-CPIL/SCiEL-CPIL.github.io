import numpy as np
import time

def construct():
    """
    Returns a sequence of step heights h in [0, 1] of length N=100.
    Sum is exactly N/2.
    Minimizes the maximum overlap of h with shifted (1-h).
    
    Strategy: 
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    This is equivalent to minimizing the maximum of the cyclic cross-correlation 
    between h and (1-h).
    
    We use Simulated Annealing with block moves on a binary-like state space,
    then fine-tune. The problem is equivalent to finding a sequence with 
    low peak sidelobe levels in its autocorrelation/cross-correlation.
    
    N is fixed to 100 as per the seed program's context, which is standard 
    for this type of discrete optimization task unless specified otherwise.
    """
    N = 100
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initialize with a binary sequence (0s and 1s) with exactly N/2 ones.
    # This is often near-optimal for such overlap minimization problems.
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Function to compute the maximum overlap with shifted complement
    # Overlap at shift k: sum_i h[i] * (1 - h[(i+k)%N])
    # Note: The task says "shifted 1-h". 
    # Let's verify the definition. 
    # "overlap of h with shifted 1-h"
    # Usually overlap at shift k is <h, T^k(1-h)> = sum h[i] (1-h[i+k]).
    # This is N/2 - <h, T^k h>.
    # Minimizing max overlap is equivalent to maximizing min <h, T^k h>.
    # Since sum h = N/2, <h, T^0 h> = sum h^2. If binary, this is N/2.
    # Overlap at 0 is N/2 - N/2 = 0? No, wait.
    # If h is binary 0/1, (1-h) is the complement.
    # Overlap at shift 0: sum h[i](1-h[i]) = 0.
    # The max overlap will be at some non-zero shift.
    
    def max_overlap(h_seq):
        # Compute all cyclic cross-correlations with (1-h)
        # Using FFT for speed
        H = np.fft.fft(h_seq)
        # Cross correlation with (1-h) is equivalent to 
        # <h, T^k(1-h)> = <h, 1> - <h, T^k h>
        # <h, 1> = sum(h) = N/2
        # So we need to compute autocorrelation of h, take min, and subtract from N/2.
        # Autocorr R[k] = sum h[i] h[i+k]
        # Max Overlap = max_k (N/2 - R[k]) = N/2 - min_k R[k]
        
        # Compute autocorrelation via FFT
        # R = ifft(fft(h) * conj(fft(h)))
        power = np.abs(H)**2
        R = np.fft.ifft(power).real
        # R[k] corresponds to shift k
        min_R = np.min(R)
        return N/2.0 - min_R

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.95
    
    best_h = h.copy()
    best_score = max_overlap(h)
    
    current_h = h.copy()
    current_score = best_score
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Perform a move: flip a block of bits
        # To maintain sum = N/2, we must swap a block of 1s with a block of 0s
        # Or simply pick two indices i, j where h[i]!=h[j] and swap them.
        # Swapping single bits is too local. Let's swap blocks.
        
        # Pick a random length for the block to swap
        # We need to find a contiguous block of 1s and a contiguous block of 0s of the same length?
        # No, we can just pick two random segments of length L, one containing mostly 1s, one mostly 0s?
        # Simpler: Pick two random indices i, j. 
        # To ensure we don't break the sum constraint easily, let's just do single bit swaps 
        # but with a "kick" of swapping multiple bits if possible.
        
        # Efficient move: Pick two random indices i and j.
        # If h[i] != h[j], swap them. This preserves the sum.
        # To make it more global, we can do this in parallel for several pairs.
        
        num_swaps = 10
        i_idx = np.random.randint(0, N, num_swaps)
        j_idx = np.random.randint(0, N, num_swaps)
        
        new_h = current_h.copy()
        # Ensure we don't swap same index or same value (which would be no-op)
        valid = (i_idx != j_idx) & (new_h[i_idx] != new_h[j_idx])
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            # Perform swaps
            # Note: if i and j are swapped, we must be careful if they overlap in the arrays
            # But since we copy first, it's safe.
            temp = new_h[i_v].copy()
            new_h[i_v] = new_h[j_v]
            new_h[j_v] = temp
        
        new_score = max_overlap(new_h)
        
        # Acceptance criterion
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_h = new_h
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        T *= alpha
    
    # Fine-tuning: Local search on the best solution found
    # Try single bit swaps to see if we can improve further
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i+1, N):
                if best_h[i] != best_h[j]:
                    new_h = best_h.copy()
                    new_h[i], new_h[j] = best_h[j], best_h[i]
                    new_score = max_overlap(new_h)
                    if new_score < best_score:
                        best_score = new_score
                        best_h = new_h
                        improved = True
    
    return best_h

def _max_overlap(h):
    N = len(h)
    H = np.fft.fft(h)
    power = np.abs(H)**2
    A = np.fft.ifft(power).real
    min_A = np.min(A)
    max_overlap = N / 2.0 - min_A
    return max_overlap
