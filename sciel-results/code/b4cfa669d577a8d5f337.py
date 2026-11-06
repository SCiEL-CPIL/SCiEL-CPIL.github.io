import math
import numpy as np
import time
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    n = 30
    
    # Calculate LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Suffix sums R[i] = sum(W[i-1] ... W[n-1])
    R = [0] * (n + 2)
    for i in range(n, 0, -1):
        R[i] = R[i + 1] + W[i - 1]
    
    # Split into two halves: 1-15 and 16-30
    mid = 15
    
    # Beam search to generate pruned sums for each half
    # Keep only top-K sums closest to 0 at each step
    K = 20000  # Increased beam width to find more diverse candidates
    
    def beam_generate(weights, K, deadline):
        """Generate pruned set of sums using beam search."""
        sums = [(0, [0] * len(weights))]  # (sum_value, coeffs)
        for i, w in enumerate(weights):
            new_sums = []
            for s, coeffs in sums:
                for c in [-1, 0, 1]:
                    new_s = s + c * w
                    new_coeffs = coeffs[:]
                    new_coeffs[i] = c
                    new_sums.append((new_s, new_coeffs))
            
            # Sort by absolute value and keep top K
            new_sums.sort(key=lambda x: abs(x[0]))
            sums = new_sums[:K]
            
            if time.time() > deadline:
                break
        
        return sums
    
    # Generate beam sums for first half (indices 0 to mid-1)
    weights_a = W[:mid]
    beam_a = beam_generate(weights_a, K, deadline)
    
    if time.time() > deadline:
        # Fallback to simple greedy
        return _greedy_fallback(W, n)
    
    # Generate beam sums for second half (indices mid to n-1)
    weights_b = W[mid:]
    beam_b = beam_generate(weights_b, K, deadline)
    
    if time.time() > deadline:
        return _greedy_fallback(W, n)
    
    # Extract just the sum values for efficient matching
    sums_a_vals = [s for s, _ in beam_a]
    sums_b_vals = [s for s, _ in beam_b]
    
    # Sort B for binary search
    sorted_b = sorted(enumerate(sums_b_vals), key=lambda x: x[1])
    arr_b_vals = np.array([x[1] for x in sorted_b])
    arr_b_idx = np.array([x[0] for x in sorted_b])
    
    # We will collect the top M candidates from the meet-in-the-middle
    # and run local search on each to find the global minimum.
    M_CANDIDATES = 20
    candidate_list = [] # (abs_val, a_coeffs, b_coeffs)
    
    # Also track greedy solution as initial best
    coeffs_greedy = _greedy(W, n)
    total_greedy = sum(c * W[i] for i, c in enumerate(coeffs_greedy))
    if total_greedy != 0:
        candidate_list.append((abs(total_greedy), coeffs_greedy[:mid], coeffs_greedy[mid:]))
    
    # Precompute max abs sum for B for pruning
    max_b_abs = max((abs(s) for s in sums_b_vals), default=0)
    
    # Meet in the middle: for each sum in A, find best matching sum in B
    for a_sum, a_coeffs in beam_a:
        if time.time() > deadline - 10: # Leave 10s for local search
            break
        
        # Pruning
        if candidate_list:
            current_best_val = min(c[0] for c in candidate_list)
            if abs(a_sum) - max_b_abs >= current_best_val:
                continue
        
        # Find b closest to -a_sum
        target = -a_sum
        idx = np.searchsorted(arr_b_vals, target)
        
        candidates = []
        if idx < len(arr_b_vals):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        
        for ci in candidates:
            b_val = arr_b_vals[ci]
            b_idx = arr_b_idx[ci]
            val = a_sum + b_val
            if val != 0:
                abs_val = abs(val)
                # Add to candidate list if it's among the best M
                if len(candidate_list) < M_CANDIDATES:
                    candidate_list.append((abs_val, a_coeffs, beam_b[b_idx][1]))
                    candidate_list.sort(key=lambda x: x[0])
                    candidate_list = candidate_list[:M_CANDIDATES]
                else:
                    if abs_val < candidate_list[-1][0]:
                        candidate_list[-1] = (abs_val, a_coeffs, beam_b[b_idx][1])
                        candidate_list.sort(key=lambda x: x[0])
    
    # Run local search on the top candidates
    best_final_coeffs = None
    best_final_val = float('inf')
    
    # Sort candidates by value to try the best ones first
    candidate_list.sort(key=lambda x: x[0])
    
    for val, a_coeffs, b_coeffs in candidate_list:
        if time.time() > deadline - 2:
            break
        final_coeffs = a_coeffs + b_coeffs
        # Local search refinement
        refined_coeffs = _local_search(final_coeffs, W, n, deadline)
        total = sum(c * W[i] for i, c in enumerate(refined_coeffs))
        if total != 0:
            abs_total = abs(total)
            if abs_total < best_final_val:
                best_final_val = abs_total
                best_final_coeffs = refined_coeffs
    
    if best_final_coeffs is not None:
        return best_final_coeffs
    
    # Fallback
    return _greedy_fallback(W, n)

def _greedy(W, n):
    """Simple greedy: pick coefficient to minimize |running sum| at each step."""
    coeffs = [0] * n
    curr_sum = 0
    for i in range(n, 0, -1):
        w = W[i - 1]
        best_c = 0
        best_val_local = curr_sum
        for c in [-1, 0, 1]:
            val = curr_sum + c * w
            if abs(val) < abs(best_val_local):
                best_val_local = val
                best_c = c
        coeffs[i - 1] = best_c
        curr_sum = best_val_local
    return coeffs

def _greedy_fallback(W, n):
    """Ensure non-zero result from greedy."""
    coeffs = _greedy(W, n)
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    if total == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
    return coeffs

def _local_search(coeffs, W, n, deadline):
    """Try flipping individual coefficients to reduce |sum|."""
    coeffs = coeffs[:] # Copy to avoid modifying original
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    if total == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
        total = sum(c * W[i] for i, c in enumerate(coeffs))
    
    improved = True
    # Add a limit to iterations to prevent infinite loops or excessive time in one candidate
    max_iter = 1000 
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(n):
            for c_new in [-1, 0, 1]:
                if c_new == coeffs[i]:
                    continue
                new_total = total + (c_new - coeffs[i]) * W[i]
                if new_total != 0 and abs(new_total) < abs(total):
                    coeffs[i] = c_new
                    total = new_total
                    improved = True
                    break
            if improved:
                break
    return coeffs
