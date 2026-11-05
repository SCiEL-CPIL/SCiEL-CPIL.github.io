"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Uses vectorized NumPy operations grouped by step size.
    """
    n = len(signs)
    max_disc = 0
    
    for d in range(1, n):
        # For step d, there are d residue classes.
        # Max length of any subsequence is ceil(n/d)
        max_len = (n - 1) // d + 1
        
        # Create a matrix to hold the subsequences
        # Shape: (d, max_len)
        # Initialize with 0s for padding
        mat = np.zeros((d, max_len), dtype=np.int8)
        
        # Fill the matrix
        # For each residue r in 0..d-1
        # mat[r, :] = signs[r::d]
        # We can do this with advanced indexing or a loop.
        # A loop of d is fine since sum(d) is N^2/2 but here we are inside the d loop.
        # Actually, we can vectorize the fill too?
        # indices = np.arange(d)[:, None] + np.arange(max_len)[None, :] * d
        # valid = indices < n
        # This might be slower due to memory allocation and masking.
        # Let's use the slice assignment in a loop, it's fast enough for small d.
        # Or better:
        # We can construct the indices for all r at once.
        # But let's stick to the simple loop for filling, as d is small on average?
        # No, d goes up to 255.
        
        # Optimized fill:
        # We know that for a given d, the elements are signs[r + k*d]
        # We can create an array of all indices:
        # idx = np.arange(d)[:, None] + np.arange(max_len)[None, :] * d
        # Then mask out invalid indices (>= n)
        # And use np.take
        # This allocates a (d, max_len) array of indices.
        
        # Let's try the loop first, it's simpler and likely fast enough given the vectorized cumsum.
        for r in range(d):
            sub = signs[r:n:d]
            mat[r, :len(sub)] = sub
            
        # Compute cumulative sums along the subsequence axis
        cum = np.cumsum(mat, axis=1)
        
        # Compute max and min for each row (each residue class)
        # The discrepancy for this residue class is max(cum) - min(cum)
        # Note: cum includes the 0th element?
        # np.cumsum does not include a leading 0.
        # The subarray sum is max(prefix) - min(prefix).
        # The prefix sums are 0, s1, s1+s2, ...
        # So we should prepend 0 to the cumsum.
        # Alternatively, just take max/min of cum and include 0?
        # If we don't prepend 0, we miss the empty subarray (sum 0).
        # But the max subarray sum is always >= 0? No, we want absolute sum.
        # The max absolute sum is max(max_subarray_sum, -min_subarray_sum).
        # max_subarray_sum = max_{i<j} (P[j] - P[i]).
        # This is equivalent to max(P) - min(P) if we include P[0]=0.
        # So we must include 0 in the set of prefix sums.
        
        # Prepend 0 to cum
        # cum is (d, max_len)
        # We can just compute max/min on cum and also consider 0?
        # Actually, the min prefix could be 0 and max prefix could be positive.
        # Or min prefix negative and max prefix 0.
        # So we need min(0, min(cum)) and max(0, max(cum)).
        
        row_maxs = np.max(cum, axis=1)
        row_mins = np.min(cum, axis=1)
        
        # The global max/min for this row considering 0
        # effective_max = max(row_maxs, 0) -> but row_maxs might be negative?
        # If all sums are negative, max is negative. But we can choose empty subarray?
        # The problem says "signed sum on ANY arithmetic progression ... including one".
        # It does not explicitly say "non-empty". But usually discrepancy is over non-empty.
        # However, if we allow empty, the sum is 0, which doesn't increase the max abs sum.
        # So we can just take max(row_maxs) and min(row_mins).
        # Wait, if the subarray is empty, sum is 0.
        # If all subarrays have negative sum, the max sum is 0 (empty).
        # But we want MAX ABSOLUTE sum.
        # So we need max( max_sum, -min_sum ).
        # max_sum = max(0, max(cum) - min(cum)) ? No.
        # max_sum = max_{i<j} (P[j] - P[i]).
        # If we include P[0]=0, then max_sum is correct.
        # Let's just prepend 0.
        
        # It's faster to just compute:
        # max_val = np.maximum(row_maxs, 0) # Not quite, if cum has positive, max is there.
        # Actually, simply:
        # max_abs_for_row = max(row_maxs - row_mins) ?
        # No. max(P) - min(P) is the max subarray sum IF 0 is included in P.
        # Since P[0]=0 is always a prefix, we just need to ensure 0 is considered.
        # If min(cum) < 0 and max(cum) > 0, then max-min is correct.
        # If min(cum) > 0, then min(P) = 0, so max_sum = max(cum) - 0 = max(cum).
        # If max(cum) < 0, then max(P) = 0, so max_sum = 0 - min(cum) = -min(cum).
        # So max_sum = max(row_maxs, 0) - min(row_mins, 0).
        
        eff_maxs = np.maximum(row_maxs, 0)
        eff_mins = np.minimum(row_mins, 0)
        
        row_discs = eff_maxs - eff_mins
        current_max = np.max(row_discs)
        
        if current_max > max_disc:
            max_disc = current_max
    
    return int(max_disc)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 256
    rng = random.Random(42)
    
    # Generate several initial candidates
    candidates = []
    
    # Random
    signs_rand = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    candidates.append(signs_rand)
    
    # Thue-Morse
    signs_tm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        t = 0
        x = i
        while x:
            t ^= (x & 1)
            x >>= 1
        signs_tm[i] = 1 if t == 0 else -1
    candidates.append(signs_tm)
    
    # Alternating
    signs_alt = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    candidates.append(signs_alt)
    
    # More random starts
    for seed in [123, 456, 789, 101, 202, 303]:
        r2 = random.Random(seed)
        candidates.append(np.array([r2.choice((-1, 1)) for _ in range(n)], dtype=np.int8))
        
    # Evaluate all candidates and pick the best
    best_signs = None
    best_disc = float('inf')
    for c in candidates:
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c.copy()
    
    signs = best_signs.copy()
    current_disc = best_disc
    
    # Hill climbing with simulated annealing
    # Since evaluation is fast, we can do many steps.
    max_rounds = 5000
    round_count = 0
    no_improve_streak = 0
    max_no_improve = 50
    
    # Temperature for simulated annealing
    temp = 1.0
    cool_rate = 0.99
    
    while time.time() < deadline and round_count < max_rounds:
        round_count += 1
        improved_this_round = False
        
        # Try flipping each position in random order
        order = np.random.default_rng(round_count * 42 + 1).permutation(n)
        
        for i in order:
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            signs[i] = -old_val
            
            new_disc = _max_ap_discrepancy(signs)
            
            if new_disc < current_disc:
                current_disc = new_disc
                best_signs = signs.copy()
                best_disc = current_disc
                improved_this_round = True
                no_improve_streak = 0
            else:
                # Simulated annealing: accept worse with probability exp(-delta/T)
                delta = new_disc - current_disc
                if delta > 0:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        # Accept worse
                        current_disc = new_disc
                        # Do not update best_signs
                        improved_this_round = True # Count as exploration step
                    else:
                        signs[i] = old_val
                else:
                    # Same or better (handled above)
                    signs[i] = old_val
        
        if improved_this_round:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            
        # Cool down
        temp *= cool_rate
        
        # If stuck, restart from best
        if no_improve_streak > max_no_improve:
            signs = best_signs.copy()
            current_disc = best_disc
            # Perturb
            num_flips = rng.randint(5, 15)
            for _ in range(num_flips):
                pos = rng.randint(0, n - 1)
                signs[pos] = -signs[pos]
            current_disc = _max_ap_discrepancy(signs)
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
            no_improve_streak = 0
            temp = 1.0 # Reset temperature
    
    # Final polish: strict hill climbing
    signs = best_signs.copy()
    current_disc = best_disc
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = signs[i]
            signs[i] = -old_val
            new_disc = _max_ap_discrepancy(signs)
            
            if new_disc < current_disc:
                current_disc = new_disc
                best_disc = current_disc
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_val
        
        if not improved:
            break
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
