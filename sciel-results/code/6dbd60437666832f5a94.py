import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Construct a large B2[2] set in 1..1000.
    Strategy:
    1. Fast greedy initialization.
    2. Local search with:
       - Pure adds (vectorized check)
       - Swaps (remove one, add one)
       - (Optional) Remove one, add two (if time allows)
    """
    deadline = time.time() + 145.0  # Safety margin
    N = 1000
    MAX_REP = 2
    
    # --- Phase 1: Greedy Initialization ---
    # We use a standard greedy: iterate 1..N, add if valid.
    # To be faster, we can just do this. It's O(N^2) but N=1000 is small.
    # However, random greedy often yields larger sets than ordered greedy for B2 sets.
    # Let's try a few random greedy seeds and keep the best.
    
    def run_greedy(seed):
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        S = []
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        
        for x in candidates:
            # Check if x can be added
            # New sums: x + s for s in S, and 2*x
            # We need counts[sum] + 1 <= MAX_REP for all new sums
            
            if not S:
                if counts[2*x] + 1 <= MAX_REP:
                    S.append(x)
                    counts[2*x] += 1
            else:
                S_arr = np.array(S)
                new_sums = S_arr + x
                new_sums = np.append(new_sums, 2*x)
                
                # Check max count
                if np.max(counts[new_sums] + 1) <= MAX_REP:
                    S.append(x)
                    counts[new_sums] += 1
                    
        return S, counts

    best_S = []
    best_counts = np.zeros(2 * N + 1, dtype=np.int32)
    best_size = 0
    
    # Try a few greedy seeds
    for seed in [0, 1, 42, 123]:
        if time.time() > deadline - 10:
            break
        S_g, c_g = run_greedy(seed)
        if len(S_g) > best_size:
            best_size = len(S_g)
            best_S = S_g
            best_counts = c_g
            
    S = best_S
    counts = best_counts
    S_set = set(S)
    outside = [x for x in range(1, N + 1) if x not in S_set]
    
    if not S:
        return S

    # --- Phase 2: Local Search ---
    
    def try_add(x, S_list, counts_arr):
        """
        Check if x can be added to S_list with current counts.
        Returns (feasible, new_sums)
        """
        if not S_list:
            new_sums = np.array([2*x], dtype=np.int64)
        else:
            S_arr = np.array(S_list)
            new_sums = S_arr + x
            new_sums = np.append(new_sums, 2*x)
            
        if np.max(counts_arr[new_sums] + 1) <= MAX_REP:
            return True, new_sums
        return False, new_sums

    def rebuild_counts(S_list):
        """Rebuild counts array from scratch for S_list."""
        c = np.zeros(2 * N + 1, dtype=np.int32)
        n = len(S_list)
        if n == 0:
            return c
        S_arr = np.array(S_list)
        ii, jj = np.triu_indices(n)
        sums = S_arr[ii] + S_arr[jj]
        c = np.bincount(sums, minlength=2 * N + 1).astype(np.int32)
        return c

    improved = True
    rng = random.Random(999)
    
    while improved and time.time() < deadline:
        improved = False
        
        # --- Phase 2a: Pure Adds ---
        if outside:
            # Vectorized check for all outside candidates
            # This might be memory heavy if outside is large and S is large.
            # |S| ~ 300, |outside| ~ 700. 300*700 = 210k sums. Manageable.
            out_arr = np.array(outside)
            S_arr = np.array(S)
            
            # Create matrix of sums: out_arr[:, None] + S_arr[None, :]
            # Plus 2*out_arr[:, None]
            cross_sums = out_arr[:, None] + S_arr[None, :]
            diag_sums = 2 * out_arr[:, None]
            all_new_sums = np.concatenate([cross_sums, diag_sums], axis=1)
            
            # Gather current counts for these sums
            gathered_counts = counts[all_new_sums]
            
            # Check feasibility: max(count + 1) <= MAX_REP
            # max_new_count = gathered_counts.max(axis=1) + 1
            max_new_count = gathered_counts.max(axis=1) + 1
            feasible_indices = np.where(max_new_count <= MAX_REP)[0]
            
            if feasible_indices.size > 0:
                # Choose the best candidate.
                # Heuristic: Minimize the maximum count reached, or minimize the sum of counts?
                # Let's minimize the max count, tie-break by minimizing the mean count.
                # Actually, minimizing the max count is primary.
                # Among feasible, pick the one with lowest max_new_count.
                # If tie, lowest mean_new_count.
                
                max_counts_for_feasible = max_new_count[feasible_indices]
                min_max_count = np.min(max_counts_for_feasible)
                
                # Filter to those with min_max_count
                candidates_with_min_max = feasible_indices[max_counts_for_feasible == min_max_count]
                
                if candidates_with_min_max.size > 1:
                    # Tie-break: minimize mean of new counts
                    mean_counts_for_candidates = gathered_counts[candidates_with_min_max].mean(axis=1)
                    best_idx_local = np.argmin(mean_counts_for_candidates)
                else:
                    best_idx_local = 0
                    
                best_idx = candidates_with_min_max[best_idx_local]
                x = int(out_arr[best_idx])
                
                # Perform the add
                S.append(x)
                S_set.add(x)
                outside.remove(x)
                
                # Update counts
                new_sums = all_new_sums[best_idx]
                counts[new_sums] += 1
                
                improved = True
                continue
        
        # --- Phase 2b: Swaps ---
        # If no pure add, try swapping an element from S with an element from outside.
        # To avoid O(|S| * |outside|) full rebuilds, we can be smarter, but for N=1000,
        # rebuilding counts for |S|-1 elements is fast if we do it efficiently.
        # However, doing this for every pair is too slow.
        # Strategy: Randomly sample a few elements from S to remove, and check if any outside fits.
        
        if outside and len(S) > 0:
            swap_attempts = 0
            max_swap_attempts = 50  # Limit per outer loop iteration
            
            while swap_attempts < max_swap_attempts and time.time() < deadline:
                swap_attempts += 1
                
                # Pick a random element to remove
                rem_idx = rng.randrange(len(S))
                rem_val = S[rem_idx]
                
                # Create temporary S without rem_val
                # For efficiency, we can just rebuild counts for S \ {rem_val}
                # But rebuilding is O(|S|^2). |S| ~ 300. 300^2 = 90k ops.
                # Doing this 50 times is 4.5M ops. Fast in NumPy.
                
                S_temp = [S[i] for i in range(len(S)) if i != rem_idx]
                counts_temp = rebuild_counts(S_temp)
                
                # Check if any outside element can be added to S_temp
                out_arr = np.array(outside)
                S_temp_arr = np.array(S_temp)
                
                if len(S_temp) == 0:
                    cross_sums = np.empty((0, 0))
                    diag_sums = 2 * out_arr[:, None]
                    all_new_sums = diag_sums
                else:
                    cross_sums = out_arr[:, None] + S_temp_arr[None, :]
                    diag_sums = 2 * out_arr[:, None]
                    all_new_sums = np.concatenate([cross_sums, diag_sums], axis=1)
                    
                gathered_counts = counts_temp[all_new_sums]
                max_new_count = gathered_counts.max(axis=1) + 1
                feasible_indices = np.where(max_new_count <= MAX_REP)[0]
                
                if feasible_indices.size > 0:
                    # Found a valid swap!
                    # Pick the best one (same heuristic as before)
                    max_counts_for_feasible = max_new_count[feasible_indices]
                    min_max_count = np.min(max_counts_for_feasible)
                    candidates_with_min_max = feasible_indices[max_counts_for_feasible == min_max_count]
                    
                    if candidates_with_min_max.size > 1:
                        mean_counts_for_candidates = gathered_counts[candidates_with_min_max].mean(axis=1)
                        best_idx_local = np.argmin(mean_counts_for_candidates)
                    else:
                        best_idx_local = 0
                        
                    best_idx = candidates_with_min_max[best_idx_local]
                    x = int(out_arr[best_idx])
                    
                    # Perform the swap
                    # Remove rem_val from S
                    S.pop(rem_idx)
                    S_set.remove(rem_val)
                    
                    # Add x to S
                    S.append(x)
                    S_set.add(x)
                    
                    # Remove x from outside
                    outside.remove(x)
                    
                    # Update counts
                    # We already have counts_temp for S \ {rem_val}
                    # We need to add x to counts_temp
                    new_sums = all_new_sums[best_idx]
                    counts_temp[new_sums] += 1
                    
                    # Set counts to counts_temp
                    counts = counts_temp
                    S_arr = np.array(S) # Update reference if needed, though we use S list mostly
                    
                    improved = True
                    break # Break out of swap attempts loop, go to next outer loop iteration
                
            # If no swap found, improved remains False (unless add found earlier)
        
        # --- Phase 2c: Remove One, Add Two (Net +1) ---
        # This is more complex. If we have time, we can try it.
        # Given the deadline, and that swaps are already O(|S|) rebuilds, 
        # double adds are O(|S| * |outside|^2) checks if done naively.
        # We'll skip this for now to ensure stability and speed, 
        # as pure adds and swaps usually suffice for good B2[2] sizes.
        
    return S

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"First 20: {result[:20]}")
