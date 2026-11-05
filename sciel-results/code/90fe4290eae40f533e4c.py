import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    # We search for the maximum k such that there exists a subset S of size k
    # where all subset sums of reciprocals are distinct.
    
    # State for backtracking:
    # current_set: list of integers in S
    # current_lcm: LCM of elements in current_set
    # sums: set of integer sums { sum_{x in T} (current_lcm / x) for T subset of current_set }
    
    def is_valid_extension(current_sums, current_lcm, x):
        """
        Checks if adding x to the current set maintains the distinct subset sums property.
        Returns (new_sums, new_lcm) if valid, else (None, None).
        """
        new_lcm = math.lcm(current_lcm, x)
        scale = new_lcm // current_lcm
        
        # Scale existing sums
        new_sums_base = {s * scale for s in current_sums}
        
        # Value for x in the new scale
        v_x = new_lcm // x
        
        # Check for collision:
        # New sums are new_sums_base U {s + v_x for s in new_sums_base}
        # We need new_sums_base to be disjoint from {s + v_x for s in new_sums_base}
        # i.e., no a, b in new_sums_base such that a = b + v_x => a - b = v_x
        
        # Optimization: Instead of building the full shifted set and checking intersection,
        # we can check if any (s + v_x) is in new_sums_base.
        # If (s + v_x) is in new_sums_base, then there exists t in new_sums_base such that t = s + v_x.
        # This means t - s = v_x.
        # This implies a collision: subset sum t (from old elements) equals subset sum s+v_x (from old elements + x).
        
        for s in new_sums_base:
            if (s + v_x) in new_sums_base:
                return None, None
        
        # If no collision, construct the new set of sums
        # new_sums = new_sums_base U (new_sums_base + v_x)
        new_sums = set(new_sums_base)
        for s in new_sums_base:
            new_sums.add(s + v_x)
            
        return new_sums, new_lcm

    def backtrack(current_set, current_lcm, sums, available, target_k, time_limit):
        if time.time() - start_time > time_limit:
            return None
        
        if len(current_set) == target_k:
            return current_set
        
        if len(current_set) + len(available) < target_k:
            return None
        
        for i, x in enumerate(available):
            new_sums, new_lcm = is_valid_extension(sums, current_lcm, x)
            if new_sums is not None:
                result = backtrack(current_set + [x], new_lcm, new_sums, available[i+1:], target_k, time_limit)
                if result is not None:
                    return result
            # If invalid, skip x. 
            # Note: We don't prune other branches here because excluding x might allow a different set.
            # However, if including x fails, we just don't include it in this branch.
            
        return None

    # Try to find the largest k
    best_result = [1] # Fallback, should not be reached if logic is correct for k=1
    
    # We iterate k from N down to 1
    # To speed up, we can start from a known good lower bound?
    # But we want MAX. So we must start high.
    
    # Heuristic ordering:
    # Sorting candidates in descending order might help find valid sets faster?
    # Or ascending?
    # Let's try descending first.
    candidates_desc = sorted(candidates, reverse=True)
    
    # To be robust, we might want to try multiple orderings if the first one fails quickly for high k?
    # But backtracking is exhaustive for the given order.
    # If a solution exists, it will be found.
    # The risk is time-out if the search space is explored inefficiently.
    
    # A better heuristic: 
    # Powers of 2 are always valid. 
    # We can try to extend the powers of 2?
    # No, just brute force with pruning is fine for N=22.
    
    for k in range(N, 0, -1):
        # Try to find a set of size k
        # Use descending order
        result = backtrack([], 1, {0}, candidates_desc, k, timeout)
        if result is not None:
            return result
        
        # If descending order didn't find it quickly, maybe try ascending?
        # Actually, if a set of size k exists, it should be found by exhaustive search.
        # If it times out, we might miss it.
        # Let's add a secondary search with ascending order if descending fails or times out?
        # But we return immediately on success.
        
        # If the loop finishes without returning, it means no set of size k was found.
        # We continue to k-1.
        
    return best_result

# EVOLVE-BLOCK-END
