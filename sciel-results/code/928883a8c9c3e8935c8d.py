import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # 1. Precompute APs
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        for s in range(n):
            idxs = list(range(s, n, d))
            if len(idxs) >= 1:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    ap_index_arrays = [np.array(idxs, dtype=np.int32) for idxs in aps]
    
    # 2. Helper functions for state management
    def compute_initial_state(signs):
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for k in range(num_aps):
            ap_sums[k] = np.sum(signs[ap_index_arrays[k]])
        abs_sums = np.abs(ap_sums)
        current_max = int(np.max(abs_sums))
        # Build counts
        counts = np.zeros(current_max + 3, dtype=np.int32)
        for s in abs_sums:
            counts[s] += 1
        return ap_sums, current_max, counts

    def get_new_max(counts, current_max, old_abs, new_abs):
        # This logic is tricky to do incrementally without knowing all changes at once.
        # Since we flip one index, we update a set of APs.
        # We will just recalculate max if counts[current_max] becomes 0, or if new_abs > current_max.
        pass

    # 3. Main Optimization Loop (Iterated Local Search)
    
    # Initial solution: Thue-Morse
    # Thue-Morse: s_i = -1 if popcount(i) is odd, 1 if even?
    # Standard Thue-Morse is 0,1,1,0... mapped to 1,-1,-1,1...
    # Let's just use a random init and let ILS do the work, or Thue-Morse as it's known to be low discrepancy.
    initial_signs = np.array([(-1) ** (bin(i).count('1') % 2) for i in range(n)], dtype=np.int8)
    
    best_signs = initial_signs.copy()
    best_max = float('inf')
    
    rng = random.Random(42)
    
    # ILS Parameters
    max_iter = 200 # Number of ILS restarts
    sa_steps_per_iter = 5000 # Steps per SA phase
    perturbation_size = 10 # Bits to flip for perturbation
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize state for this ILS iteration
        # If it==0, use Thue-Morse. Else, perturb best_signs.
        if it == 0:
            signs = initial_signs.copy()
        else:
            signs = best_signs.copy()
            for _ in range(perturbation_size):
                idx = rng.randrange(n)
                signs[idx] = -signs[idx]
                
        ap_sums, current_max, counts = compute_initial_state(signs)
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            
        # Simulated Annealing Phase
        T = 5.0
        cooling_rate = 0.9995
        min_T = 0.01
        no_improve = 0
        max_no_improve = 2000
        
        while True:
            if time.time() - start_time > time_limit:
                break
            if T < min_T or no_improve > max_no_improve:
                break
                
            # Pick random index to flip
            i = rng.randrange(n)
            old_sign = signs[i]
            delta = -2 * old_sign
            
            affected_aps = index_to_aps[i]
            if not affected_aps:
                continue
            
            # Calculate new sums for affected APs
            old_vals = ap_sums[affected_aps]
            new_vals = old_vals + delta
            
            # Update counts and determine new max
            # We need to be careful with counts array size.
            # Max possible sum is n=256. So counts size 257 is enough.
            
            # Temporarily update counts
            old_abs_vals = np.abs(old_vals)
            new_abs_vals = np.abs(new_vals)
            
            # Check if current_max changes
            # If any new_abs > current_max, new_max is max(new_abs)
            # Else if count of current_max becomes 0, find next lower max.
            
            # Optimization: Only do full max recalculation if necessary.
            # For simplicity and robustness in Python, let's just track the max.
            # Since num_aps is ~32k, and we do ~5k steps, this might be slow if we recompute max every time.
            # But we only recompute if `had_max` logic suggests it.
            
            # Let's use the counts array properly.
            # Ensure counts array is large enough (size 257)
            if len(counts) < 257:
                counts = np.pad(counts, (0, 257 - len(counts)))
            
            max_new_abs = int(np.max(new_abs_vals))
            
            # Update counts
            # We subtract old, add new
            # Vectorized update for counts is hard with indices, so loop affected APs (small number, usually < 100)
            for k in affected_aps:
                old_a = abs(ap_sums[k])
                new_a = abs(ap_sums[k] + delta)
                counts[old_a] -= 1
                counts[new_a] += 1
                ap_sums[k] += delta # Update actual sum
            
            # Determine new_max
            if max_new_abs > current_max:
                new_max = max_new_abs
            else:
                # Check if current_max is still populated
                if counts[current_max] == 0:
                    # Find new max
                    # It must be < current_max
                    # Scan down from current_max - 1
                    new_max = current_max - 1
                    while new_max >= 0 and counts[new_max] == 0:
                        new_max -= 1
                    if new_max < 0:
                        new_max = 0 # Should not happen
                else:
                    new_max = current_max
            
            cost_diff = new_max - current_max
            
            # Acceptance
            accept = False
            if cost_diff < 0:
                accept = True
            elif cost_diff == 0:
                accept = True # Or random
            else:
                if T < 1e-5:
                    accept = False
                else:
                    prob = math.exp(-cost_diff / T)
                    accept = rng.random() < prob
            
            if accept:
                signs[i] = -signs[i]
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert counts and ap_sums
                for k in affected_aps:
                    new_a = abs(ap_sums[k])
                    old_a = abs(ap_sums[k] - delta) # Since we added delta, subtract to revert
                    counts[new_a] -= 1
                    counts[old_a] += 1
                    ap_sums[k] -= delta
                # Revert sign
                signs[i] = old_sign
                no_improve += 1
                
            T *= cooling_rate

    return best_signs.tolist()
