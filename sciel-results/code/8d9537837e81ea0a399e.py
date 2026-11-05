"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    # We iterate over step sizes d.
    # For a given d, the progressions are determined by start index i (0 <= i < d) 
    # and length L.
    # An AP starting at i with step d is s[i], s[i+d], s[i+2d], ...
    # We want max |sum_{k=0}^{L-1} s[i + k*d]| for all valid i, L.
    
    # Optimization:
    # For a fixed d, we can process all starting indices i in [0, d) simultaneously.
    # Let's create a 2D array A where A[i, k] = s[i + k*d] for all valid i, k.
    # The dimensions are d x (n//d + 1) approximately.
    # Actually, for a fixed d, the valid indices for start i are 0..d-1.
    # The length of the progression for start i is floor((n-1-i)/d) + 1.
    
    # A more efficient way:
    # For each d from 1 to n//2 (since d > n//2 implies length 1, max sum is 1):
    #   Actually, for d > n//2, the only APs are singletons (length 1) or maybe length 2?
    #   If d > n//2, then i+d >= n for most i. 
    #   Specifically, if d >= n, no AP of length >= 2.
    #   If d > n//2, then for any i, i+d < n implies i < n-d < n/2.
    #   So only a few starts have length >= 2.
    #   However, the max absolute sum for any AP is at least 1 (singleton).
    #   We are looking for the *maximum* absolute sum.
    #   Singletons always give sum +/- 1.
    #   So we only need to check d where APs of length >= 2 exist, i.e., d <= n-1.
    #   But for large d, the number of such APs is small.
    
    # Let's iterate d from 1 to n-1.
    # For efficiency, we can stop early if we find a large discrepancy? No, we need the max.
    # But we can skip d where max possible length is 1. That is d >= n.
    # So d from 1 to n-1.
    
    # Vectorized approach for a fixed d:
    # We want to compute prefix sums along the "diagonal" defined by step d.
    # For start i in 0..d-1:
    #   The sequence is s[i], s[i+d], s[i+2d], ...
    #   Let len_i = floor((n-1-i)/d) + 1
    #   We compute cumulative sums of this sequence.
    #   The max abs value in this cumulative sum array is a candidate.
    
    # To vectorize over i:
    # Create an array of indices for a fixed d?
    # It's easier to just loop d, and for each d, create a 2D view if possible, or just compute.
    # n=256 is small.
    # Total operations: sum_{d=1}^{255} (n/d * d) ~ n * H_n ~ 256 * 6 ~ 1500 operations?
    # Wait, the work for fixed d is O(n) if we do it smartly.
    # For fixed d, we have d sequences. Total elements across all sequences is n.
    # So for each d, we do O(n) work.
    # Total work O(n^2) = 256^2 = 65536. This is very fast.
    
    for d in range(1, n):
        # For each start i in 0..d-1
        # We can extract the subsequence s[i::d]
        # Compute cumulative sum
        # Find max abs
        # This is O(n/d) per i, total O(n) per d.
        
        # Using numpy slicing is efficient.
        for i in range(d):
            seq = s[i::d]
            if len(seq) == 0:
                continue
            cumsum = np.cumsum(seq)
            max_val = np.max(np.abs(cumsum))
            if max_val > max_abs:
                max_abs = max_val
                # Early exit? If max_abs reaches a high value, maybe we can stop?
                # But we need the exact max for the SA cost function.
                # However, if we are just optimizing, a lower bound is fine?
                # No, we need the actual max to compare states.
                # But if max_abs is already very large, further d won't change it much?
                # Actually, we just want to minimize the max.
                # If we find max_abs = 10, and we know it can't be less than 10 for this state,
                # we don't need to check other d's if we are just comparing to a previous state?
                # No, we need the full max for the current state to evaluate it.
                # But we can break the inner loop if max_abs is already very high?
                # No, because a later d might be even higher.
                # But we can break the outer loop if max_abs is very high? No.
                
    return max_abs

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s to be safe
    
    # Initial random solution
    rng = random.Random(171)
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int16)
    
    current_cost = _max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.99
    
    # We will try to flip one sign at a time.
    # To speed up evaluation, we can't easily update the max AP discrepancy incrementally
    # because flipping one sign affects O(n) APs (those containing that index).
    # But n=256 is small, so full re-evaluation is ~65k ops, which is ~0.01s in Python/NumPy?
    # Let's check speed.
    # If full eval is 0.01s, we can do 8500 steps in 85s.
    # That should be enough for SA to find a good solution.
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.randint(0, n-1)
        signs[idx] *= -1
        
        new_cost = _max_ap_discrepancy(signs)
        
        if new_cost < current_cost:
            # Accept
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            # Reject with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() > prob:
                    # Reject: revert flip
                    signs[idx] *= -1
            else:
                # This case shouldn't happen if we check new_cost < current_cost first
                # but just in case
                current_cost = new_cost
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_signs.tolist()
# EVOLVE-BLOCK-END
