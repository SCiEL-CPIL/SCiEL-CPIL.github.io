"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Numbers of interest
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    
    # Map value to index
    idx = {x: i for i, x in enumerate(nums)}
    
    # Precompute conflict matrix
    # conflict[i] is a bitmask of indices j such that lcm(nums[i], nums[j]) <= 256
    # We use integers as bitmasks
    conflict = [0] * n
    
    for i in range(n):
        ai = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            bj = nums[j]
            if lcm(ai, bj) <= N:
                mask |= (1 << j)
        conflict[i] = mask

    # Precompute reciprocal weights
    weights = [1.0 / x for x in nums]
    
    # Helper function to check if a set of indices is valid
    def is_valid_set(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if conflict[indices[i]] & (1 << indices[j]):
                    return False
        return True
    
    # Helper function to compute score
    def compute_score(indices):
        return sum(weights[i] for i in indices)
    
    # We split the numbers into "Small" (2..16) and "Large" (17..256)
    small_vals = list(range(2, 17))
    small_indices = [idx[x] for x in small_vals]
    n_small = len(small_indices)
    
    # Precompute list of large indices
    large_indices_list = [i for i in range(n) if nums[i] >= 17]
    
    best_set_indices = []
    best_score = 0.0
    
    # Iterate over all subsets of small numbers
    # There are 2^15 = 32768 subsets
    for mask_small in range(1 << n_small):
        if time.time() > deadline:
            break
            
        # Extract small indices in this mask
        small_in_set = []
        for i in range(n_small):
            if mask_small & (1 << i):
                small_in_set.append(small_indices[i])
                
        # Check internal validity of small_in_set
        if not is_valid_set(small_in_set):
            continue
            
        # Determine compatible large numbers
        large_mask = 0
        for i in large_indices_list:
            large_mask |= (1 << i)
        
        # For each small index in small_in_set, remove its conflicts from large_mask
        for idx_i in small_in_set:
            large_mask &= ~conflict[idx_i]
            
        # Extract active large candidates
        active_candidates = [i for i in large_indices_list if large_mask & (1 << i)]
        
        # Sort by reciprocal weight (descending) to prioritize high weights
        active_candidates.sort(key=lambda x: -weights[x])
        
        # Greedy selection
        selected_large = []
        selected_mask = 0
        
        for i in active_candidates:
            if selected_mask & conflict[i]:
                continue
            selected_large.append(i)
            selected_mask |= (1 << i)
            
        # Combine small and large
        full_set_indices = small_in_set + selected_large
        
        # Calculate score
        score = compute_score(full_set_indices)
        
        if score > best_score:
            best_score = score
            best_set_indices = list(full_set_indices)
    
    # Local search with 1-for-1 swaps and multiple restarts
    if time.time() < deadline - 10.0 and best_set_indices:
        current_set = list(best_set_indices)
        current_score = best_score
        
        # Multiple restarts with random perturbations
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Create a perturbed version of the current best
            if restart == 0:
                working_set = list(current_set)
            else:
                # Randomly remove 1-3 elements and add 1-3 random compatible elements
                working_set = list(current_set)
                remove_count = min(3, len(working_set))
                if remove_count > 0:
                    remove_indices = random.sample(range(len(working_set)), remove_count)
                    for ri in sorted(remove_indices, reverse=True):
                        working_set.pop(ri)
                
                # Try to add some random compatible elements
                all_indices = list(range(n))
                random.shuffle(all_indices)
                added = 0
                for i in all_indices:
                    if added >= 3:
                        break
                    if i not in working_set and is_valid_set(working_set + [i]):
                        working_set.append(i)
                        added += 1
            
            working_score = compute_score(working_set)
            
            # Local search: try 1-for-1 swaps
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                working_set_set = set(working_set)
                
                # Try removing one and adding another
                for remove_idx in range(len(working_set)):
                    if time.time() > deadline - 5.0:
                        break
                    remove_i = working_set[remove_idx]
                    
                    # Try adding each candidate not in working_set
                    for add_i in range(n):
                        if add_i in working_set_set:
                            continue
                        # Check if add_i is compatible with all in working_set except remove_i
                        new_set = [x for x in working_set if x != remove_i]
                        if is_valid_set(new_set + [add_i]):
                            new_score = working_score - weights[remove_i] + weights[add_i]
                            if new_score > working_score + 1e-12:
                                working_set = new_set + [add_i]
                                working_set_set = set(working_set)
                                working_score = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            # Update global best if improved
            if working_score > best_score + 1e-12:
                best_score = working_score
                best_set_indices = list(working_set)
                current_set = list(working_set)
    
    # Final high-precision polish: 2-for-1 and 1-for-2 swaps
    if time.time() < deadline - 3.0 and best_set_indices:
        working_set = list(best_set_indices)
        working_score = compute_score(working_set)
        improved = True
        
        while improved and time.time() < deadline - 3.0:
            improved = False
            working_set_set = set(working_set)
            
            # Try 2-for-1 swaps: remove two, add one
            for i in range(len(working_set)):
                if time.time() > deadline - 3.0:
                    break
                for j in range(i + 1, len(working_set)):
                    if time.time() > deadline - 3.0:
                        break
                    remove_i = working_set[i]
                    remove_j = working_set[j]
                    remaining = [x for x in working_set if x != remove_i and x != remove_j]
                    
                    for add_i in range(n):
                        if add_i in working_set_set:
                            continue
                        if is_valid_set(remaining + [add_i]):
                            new_score = working_score - weights[remove_i] - weights[remove_j] + weights[add_i]
                            if new_score > working_score + 1e-12:
                                working_set = remaining + [add_i]
                                working_set_set = set(working_set)
                                working_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try 1-for-2 swaps: remove one, add two
            if not improved:
                for remove_idx in range(len(working_set)):
                    if time.time() > deadline - 3.0:
                        break
                    remove_i = working_set[remove_idx]
                    remaining = [x for x in working_set if x != remove_i]
                    
                    # Try adding two compatible elements
                    for add1 in range(n):
                        if time.time() > deadline - 3.0:
                            break
                        if add1 in working_set_set:
                            continue
                        if not is_valid_set(remaining + [add1]):
                            continue
                        
                        for add2 in range(n):
                            if time.time() > deadline - 3.0:
                                break
                            if add2 in working_set_set or add2 == add1:
                                continue
                            if not is_valid_set(remaining + [add1, add2]):
                                continue
                            
                            new_score = working_score - weights[remove_i] + weights[add1] + weights[add2]
                            if new_score > working_score + 1e-12:
                                working_set = remaining + [add1, add2]
                                working_set_set = set(working_set)
                                working_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Update global best if improved
            if working_score > best_score + 1e-12:
                best_score = working_score
                best_set_indices = list(working_set)
    
    # Convert indices back to values
    result = [nums[i] for i in best_set_indices]
    
    return result
# EVOLVE-BLOCK-END
