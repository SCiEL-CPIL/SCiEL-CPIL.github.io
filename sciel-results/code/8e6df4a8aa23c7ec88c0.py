"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s kill
    
    # The problem:
    # We have a sequence h of length N, where each h_i in [0, 1].
    # Constraint: sum(h) = N/2. (Since step heights are on [0,2] with integral 1? 
    # Wait, the task says: "Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1".
    # "sum is exactly len/2" -> sum(h) = N/2.
    # Minimize the maximum overlap of h with shifted 1-h.
    
    # What is "overlap of h with shifted 1-h"?
    # Let k be the shift. The overlap at shift k is likely sum_i h[i] * (1-h[i-k]) or similar.
    # Or is it the maximum of the cross-correlation?
    # "minimize the maximum overlap of h with shifted 1-h"
    # Let's assume the metric is: max_k sum_i h[i] * (1-h[(i-k) mod N])
    # Or maybe max_k sum_i h[i] * h[(i+k) mod N]? No, it says "overlap of h with shifted 1-h".
    # So it's the correlation between h and (1-h) shifted by k.
    # Let's define overlap(k) = sum_i h[i] * (1 - h[(i-k) % N]).
    # We want to minimize max_k overlap(k).
    
    # Let's verify with the seed: h = 0.5 everywhere.
    # sum(h) = 50.
    # overlap(k) = sum_i 0.5 * (1 - 0.5) = sum_i 0.25 = 12.5.
    # Max overlap = 12.5.
    # Score is 1.0.
    
    # If we use a binary sequence (0 and 1) with 50 ones and 50 zeros.
    # overlap(k) = sum_i h[i] * (1 - h[(i-k) % N]).
    # If h is 1, 1-h is 0. If h is 0, 1-h is 1.
    # So overlap(k) counts the number of positions where h[i]=1 and h[i-k]=0.
    # Let A be the set of indices where h[i]=1. |A|=50.
    # overlap(k) = |A| - |A ∩ (A+k)| = 50 - autocorrelation(k).
    # To minimize max overlap, we need to maximize min autocorrelation(k) for k != 0?
    # No, we want to minimize max_k (50 - autocorrelation(k)).
    # This is equivalent to maximizing min_k autocorrelation(k).
    # For a balanced binary sequence, the average autocorrelation is 0.
    # We want the autocorrelation to be as flat as possible.
    # This is related to Barker sequences or m-sequences.
    # For N=100, we can search for a good binary sequence.
    
    # Let's try to find a binary sequence with low maximum overlap.
    # We'll use a local search (simulated annealing or simple hill climbing with restarts).
    
    best_h = np.full(N, 0.5)
    best_score = float('inf')
    
    def compute_score(h):
        # h is a numpy array of length N
        # Calculate max_k sum_i h[i] * (1 - h[(i-k) % N])
        # This is equivalent to max_k (sum(h) - sum_i h[i]*h[(i-k)%N])
        # sum(h) is fixed at N/2.
        # So we want to minimize max_k (N/2 - autocorr(k))
        # Which is equivalent to maximizing min_k autocorr(k).
        # Let's compute autocorr for all shifts.
        # autocorr(k) = sum_i h[i] * h[(i-k) % N]
        # We can use FFT for speed, but N=100 is small.
        max_overlap = 0
        for k in range(1, N):
            overlap = np.sum(h * np.roll(h, k))
            # Wait, the definition was sum_i h[i] * (1 - h[(i-k) % N])
            # = sum(h) - sum_i h[i] * h[(i-k) % N]
            # = N/2 - autocorr(k)
            # So we want to minimize max_k (N/2 - autocorr(k))
            # = N/2 - min_k autocorr(k)
            # So we want to maximize min_k autocorr(k).
            # Let's just compute the overlap directly to be safe.
            shifted_1_minus_h = 1 - np.roll(h, k)
            overlap = np.sum(h * shifted_1_minus_h)
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap
    
    # Initial binary sequence
    h = np.zeros(N)
    h[:N//2] = 1
    
    # Shuffle to get a random start
    np.random.shuffle(h)
    
    current_score = compute_score(h)
    best_score = current_score
    best_h = h.copy()
    
    # Local search
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
        
        # Swap h[i] and h[j]
        h[i], h[j] = h[j], h[i]
        new_score = compute_score(h)
        
        if new_score < current_score or np.random.random() < np.exp((current_score - new_score) / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Revert
            h[i], h[j] = h[j], h[i]
        
        T *= cooling
    
    # If we have time, try a few more restarts
    restarts = 0
    while time.time() < deadline and restarts < 10:
        h = np.zeros(N)
        h[:N//2] = 1
        np.random.shuffle(h)
        current_score = compute_score(h)
        
        T = 1.0
        while time.time() < deadline and T > T_min:
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            h[i], h[j] = h[j], h[i]
            new_score = compute_score(h)
            
            if new_score < current_score or np.random.random() < np.exp((current_score - new_score) / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                h[i], h[j] = h[j], h[i]
            
            T *= cooling
        restarts += 1
    
    # Return the best binary sequence found
    # The task says "sequence of step heights in [0, 1] whose sum is exactly len/2"
    # A binary sequence with 50 ones and 50 zeros satisfies this.
    return best_h
# EVOLVE-BLOCK-END
