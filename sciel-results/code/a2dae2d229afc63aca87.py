"""
Binary Combinatorial Search Strategy for Erdos Minimum Overlap.

This strategy abandons the continuous spectral representation and instead
searches directly over the space of binary sequences (h_i in {0, 1}) with
exactly N/2 ones. This is a discrete combinatorial optimization problem.

The objective is to minimize the maximum overlap of h with shifted 1-h.
For binary h, overlap_k = sum_{i} h_i * (1 - h_{i-k}).
This is equivalent to minimizing the maximum number of "1"s in h that
overlap with "0"s in the shifted version of h.

We use a multi-start local search approach:
1. Generate random binary sequences with exactly N/2 ones.
2. Apply local moves:
   a. Single bit flip (if it doesn't violate the sum constraint? No, sum must be N/2).
      So single flips are invalid unless paired.
   b. Pair swap: Flip a 1 to 0 and a 0 to 1. This preserves the sum.
   c. We evaluate the change in the maximum overlap.
3. Accept moves that reduce the maximum overlap.
4. Use randomized restarts to escape local minima.

To speed up evaluation, we use FFT to compute the autocorrelation/overlap.
overlap_k = N/2 - autocorr_k (for binary h with sum N/2).
Actually, for binary h:
autocorr_k = sum h_i h_{i-k}.
overlap_k = sum h_i (1 - h_{i-k}) = sum h_i - sum h_i h_{i-k} = N/2 - autocorr_k.
So minimizing max overlap is equivalent to maximizing min autocorr_k.

We will implement a fast local search that:
- Maintains the binary array.
- Precomputes the current autocorrelation.
- When considering a move (swap i and j where h[i]=1, h[j]=0),
  we can update the autocorrelation efficiently or just recompute if N is small enough.
  Given N=100, recomputing autocorr via FFT is very fast (~microseconds).
  So we can afford to recompute the score for each candidate move.
"""

import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_overlap_score(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    For binary h with sum N/2:
    overlap_k = N/2 - autocorr_k
    We want to minimize max_k overlap_k.
    """
    # Compute circular autocorrelation
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # Due to numerical errors, clip to valid range
    autocorr = np.clip(autocorr, 0, N/2)
    overlaps = N/2.0 - autocorr
    return np.max(overlaps)

def _generate_random_binary(n_ones=N//2):
    """Generate a random binary sequence with exactly n_ones ones."""
    h = np.zeros(N)
    indices = np.random.choice(N, n_ones, replace=False)
    h[indices] = 1.0
    return h

def _local_search(h, max_iters=5000, time_limit=5.0):
    """
    Perform local search on a binary sequence h.
    Moves: Swap a 1 at index i with a 0 at index j.
    Accept if the new max overlap is lower than the current max overlap.
    """
    start_time = time.time()
    current_score = _compute_overlap_score(h)
    best_h = h.copy()
    best_score = current_score
    
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    # We need to maintain lists of ones and zeros to sample efficiently
    # But since we only do swaps, the sets change.
    # For N=100, we can just recompute indices or keep a set.
    # Let's keep a set of indices for ones and zeros for O(1) lookup and sampling.
    ones_set = set(ones)
    zeros_set = set(zeros)
    
    no_improve_count = 0
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        if not ones_set or not zeros_set:
            break
            
        # Randomly pick a 1 and a 0 to swap
        i = np.random.choice(list(ones_set))
        j = np.random.choice(list(zeros_set))
        
        # Try the swap
        h[i] = 0
        h[j] = 1
        ones_set.remove(i)
        ones_set.add(j)
        zeros_set.add(i)
        zeros_set.remove(j)
        
        new_score = _compute_overlap_score(h)
        
        if new_score < current_score - 1e-9:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
            no_improve_count = 0
        else:
            # Revert
            h[i] = 1
            h[j] = 0
            ones_set.add(i)
            ones_set.remove(j)
            zeros_set.remove(i)
            zeros_set.add(j)
            no_improve_count += 1
            
            # If stuck for too long, break to allow restart
            if no_improve_count > 1000:
                break
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    np.random.seed(42)
    
    best_h = _generate_random_binary()
    best_score = _compute_overlap_score(best_h)
    
    # Multi-start local search
    # We will run as many restarts as time permits.
    # Each restart does a local search from a random start.
    
    num_restarts = 0
    while time.time() < deadline:
        # Generate a new random start
        h_init = _generate_random_binary()
        
        # Run local search
        # Limit time per local search to allow multiple restarts
        # If we have lots of time, we can allow longer searches.
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed
        
        # Heuristic: if we have > 10s left, allow up to 5s per search.
        # If we have < 10s left, allow only what fits.
        max_time_for_search = min(5.0, time_remaining / 2.0) if time_remaining > 10 else time_remaining / 2.0
        
        if max_time_for_search < 0.1:
            break
            
        h_opt, score_opt = _local_search(h_init, max_iters=20000, time_limit=max_time_for_search)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
            
        num_restarts += 1
        
        # Optional: If we find a very good score, we might want to polish it further
        # but the local search already does that.
        
        # Early stopping if we hit a theoretical lower bound?
        # Theoretical lower bound for N=100 is unknown, but 0 is impossible.
        # For random binary, score is around N/4 = 25.
        # Good sequences might have score around 10-15.
        
    # Final check: ensure the returned h is valid
    # It is binary, so sum is exactly N/2.
    # Values are 0 or 1, so in [0, 1].
    
    return best_h

# EVOLVE-BLOCK-END
