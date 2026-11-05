import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def is_dissociated(S):
        """
        Check if the set S is dissociated, i.e., all subset sums of reciprocals are distinct.
        Equivalent to checking if all subset sums of {L/x for x in S} are distinct,
        where L = lcm(S).
        """
        if not S:
            return True
        
        # Calculate LCM
        L = 1
        for x in S:
            L = math.lcm(L, x)
        
        # Convert to integer values
        vals = [L // x for x in S]
        
        # Generate all subset sums
        sums = {0}
        for i, v in enumerate(vals):
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
            # Check for collision
            if len(sums) < (1 << (i + 1)):
                return False
                
        return len(sums) == (1 << len(S))

    def backtrack(current, available, target_size, time_limit):
        if time.time() - start_time > time_limit:
            return None
            
        if len(current) == target_size:
            return current
            
        if len(current) + len(available) < target_size:
            return None
            
        # Heuristic: Try smaller numbers first to maximize "density" of small reciprocals?
        # Or larger numbers?
        # Actually, for dissociated sets, it's often better to start with large numbers
        # because they are less likely to create small linear dependencies?
        # Let's try descending order (largest first) as a heuristic.
        # The available list is sorted ascending, so we iterate in reverse.
        
        # Optimization: Pre-check if remaining elements can form the target size
        # is handled by the len check above.
        
        # Iterate in reverse (largest to smallest)
        for i in range(len(available) - 1, -1, -1):
            x = available[i]
            new_current = current + [x]
            
            # Check if new_current is dissociated
            if is_dissociated(new_current):
                # Recurse with elements smaller than x (indices 0 to i-1)
                # Since we are picking in descending order, the next candidates must be < x
                # to maintain uniqueness and avoid reordering issues?
                # Actually, the set is unordered. We just need to pick distinct elements.
                # If we pick x, the next candidates should be from available[0:i] (elements < x)
                # to ensure we don't pick duplicates and to keep the search space structured.
                # This effectively enforces a sorted order of selection (descending).
                result = backtrack(new_current, available[:i], target_size, time_limit)
                if result is not None:
                    return result
            else:
                continue
                
        return None

    # Try to find the largest possible set
    for k in range(N, 0, -1):
        # Use full candidate list, sorted ascending
        available = sorted(candidates)
        
        result = backtrack([], available, k, timeout)
        
        if result is not None:
            return result
            
        if time.time() - start_time > timeout - 5:
             break

    # Fallback: Return a known valid set
    return [1, 2, 4, 8, 16]
