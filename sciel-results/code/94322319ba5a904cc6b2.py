import time
import math
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 22
    candidates = list(range(1, N + 1))
    
    # Precompute LCM and values
    L = 1
    for x in candidates:
        L = math.lcm(L, x)
        
    vals = [L // x for x in candidates]
    
    def get_new_sums(current_sums, new_val):
        """
        Returns the new set of sums if adding new_val is valid (no collisions), else None.
        """
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + new_val)
        # If there was a collision, the size will be less than 2 * len(current_sums)
        # But we can just check the size against the expected size.
        return new_sums

    def backtrack(current_indices, current_sums, available_indices, target_size):
        """
        Recursive backtracking.
        """
        if time.time() - start_time > timeout:
            return None
            
        if len(current_indices) == target_size:
            return current_indices
            
        # Pruning: not enough elements left
        if len(current_indices) + len(available_indices) < target_size:
            return None
            
        for i, idx in enumerate(available_indices):
            v = vals[idx]
            new_sums = get_new_sums(current_sums, v)
            
            # Check for collision
            # The number of sums should double.
            # If len(new_sums) < 2 * len(current_sums), there's a collision.
            # Note: len(current_sums) is 2^k.
            if len(new_sums) < 2 * len(current_sums):
                continue
                
            result = backtrack(
                current_indices + [idx],
                new_sums,
                available_indices[i+1:],
                target_size
            )
            if result is not None:
                return result
                
        return None

    best_result = None
    best_size = 0
    
    # Phase 1: Find a set of size 16.
    # This should be fast.
    # Try a few random orders.
    deadline_16 = start_time + 5.0
    for _ in range(20):
        if time.time() > deadline_16:
            break
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        result = backtrack([], {0}, shuffled_candidates, 16)
        if result is not None:
            best_result = result
            best_size = 16
            break
            
    # If we didn't find 16, try greedy as a fallback (though unlikely to fail for 16)
    if best_result is None:
        # Greedy fallback
        greedy_indices = []
        greedy_sums = {0}
        for idx in candidates:
            v = vals[idx]
            new_sums = get_new_sums(greedy_sums, v)
            if len(new_sums) == 2 * len(greedy_sums):
                greedy_indices.append(idx)
                greedy_sums = new_sums
        best_result = greedy_indices
        best_size = len(greedy_indices)

    # Phase 2: Try to find a set of size 17.
    if best_size < 17:
        deadline_17 = start_time + timeout
        # Use remaining time for 17
        # Try more random orders
        for _ in range(1000):
            if time.time() > deadline_17:
                break
            shuffled_candidates = candidates[:]
            random.shuffle(shuffled_candidates)
            result = backtrack([], {0}, shuffled_candidates, 17)
            if result is not None:
                best_result = result
                best_size = 17
                break
                
    return best_result
