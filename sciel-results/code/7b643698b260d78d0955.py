"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a set of up to 80 distinct positive integers <= 4096
    where all unordered triples (with repetition) have unique sums.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    max_val = 4096
    target_size = 80
    
    best_set = []
    
    # Strategies: 'smallest', 'largest', 'random'
    strategies = ['smallest', 'largest', 'random']
    
    # Allocate time for each strategy
    # We will try to fit at least 2-3 strategies if possible
    time_per_strategy = 15.0
    
    for strat in strategies:
        if time.time() > deadline - 2.0:  # Leave 2s for final check
            break
            
        current_deadline = time.time() + time_per_strategy
        if current_deadline > deadline:
            current_deadline = deadline
            
        # Initialize used_sums as a boolean array
        # Max sum is 3 * 4096 = 12288
        max_sum = 3 * max_val
        used_sums = np.zeros(max_sum + 1, dtype=bool)
        
        selected = []
        
        # Prepare candidate list based on strategy
        if strat == 'smallest':
            candidates = list(range(1, max_val + 1))
        elif strat == 'largest':
            candidates = list(range(max_val, 0, -1))
        else:  # random
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > current_deadline:
                break
            if len(selected) >= target_size:
                break
            
            # Generate all new sums involving x
            # 1. 3x
            new_sums = np.array([3 * x], dtype=np.int32)
            
            # 2. 2x + y for y in selected
            if selected:
                sel_arr = np.array(selected, dtype=np.int32)
                new_sums = np.concatenate([new_sums, 2 * x + sel_arr])
                
                # 3. x + 2y for y in selected
                new_sums = np.concatenate([new_sums, x + 2 * sel_arr])
                
                # 4. x + y + z for y <= z in selected
                # We need all pairs (y, z) with y <= z
                # This is combinations_with_replacement
                # For small n, we can generate indices
                n = len(selected)
                if n > 0:
                    # Generate all pairs i, j such that 0 <= i <= j < n
                    # Using meshgrid or loops is fine for n <= 80
                    # But let's try to be efficient.
                    # For n=80, n*(n+1)/2 = 3240 pairs.
                    # 3240 operations in Python is fast.
                    # But we can vectorize if we store selected as array.
                    # Let's just use a loop for pair generation to avoid complex numpy broadcasting for small n
                    # Actually, numpy broadcasting is better.
                    # Create indices
                    i_idx, j_idx = np.triu_indices(n)
                    pairs = sel_arr[i_idx] + sel_arr[j_idx]
                    new_sums = np.concatenate([new_sums, x + pairs])
            
            # Check for collisions
            # new_sums might have duplicates internally? No, because:
            # 3x is unique.
            # 2x+y and x+2y are distinct for y != z?
            # Wait, 2x+y1 = 2x+y2 implies y1=y2.
            # x+2y1 = x+2y2 implies y1=y2.
            # 2x+y1 = x+2y2 => x + y1 = 2y2. Possible collision.
            # x+y1+z1 = x+y2+z2 => y1+z1 = y2+z2. Possible collision.
            # So new_sums might contain duplicates.
            # We just need to check if ANY of these sums are in used_sums.
            # Also, we should check if any NEW sum is already in NEW sums?
            # If two different triples involving x have the same sum, that's a collision too.
            # So we need to check uniqueness among new_sums as well.
            
            # Efficient check:
            # 1. Check against used_sums
            # 2. Check internal uniqueness of new_sums
            
            # Check against used_sums
            if np.any(used_sums[new_sums]):
                continue
            
            # Check internal uniqueness of new_sums
            # We can use np.unique and check length
            # Or use a set for small n
            # np.unique is O(M log M) where M is len(new_sums) ~ 1 + 2n + n^2/2
            # For n=80, M ~ 3240. np.unique is fast.
            unique_new = np.unique(new_sums)
            if len(unique_new) != len(new_sums):
                continue
            
            # If no collisions, add x
            used_sums[new_sums] = True
            selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) == target_size:
                break
    
    # Final verification (optional, but good for safety)
    # For now, trust the logic.
    return best_set

# EVOLVE-BLOCK-END
