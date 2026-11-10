import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # LCM of 1..22
    # 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
    lcm_val = 232792560
    
    # Precompute weights
    weights = {i: lcm_val // i for i in range(1, 23)}
    
    def get_subset_sums(S):
        """Returns a set of all subset sums of weights for S."""
        if not S:
            return {0}
        sums = {0}
        for x in S:
            w = weights[x]
            # Add w to all existing sums
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        return sums
        
    def is_dissociated(S):
        """Checks if all subset sums are distinct."""
        sums = get_subset_sums(S)
        return len(sums) == (1 << len(S))
        
    # Greedy construction
    def greedy_construct():
        S = []
        remaining = set(range(1, 23))
        
        while remaining:
            best_x = None
            best_min_dist = -1
            best_sums = None
            
            for x in remaining:
                candidate_S = S + [x]
                sums = get_subset_sums(candidate_S)
                
                if len(sums) != (1 << len(candidate_S)):
                    continue # Collision, skip
                
                # Calculate min distance
                # This is expensive, so we only do it if valid
                # Actually, for greedy, we want to maximize the "safety margin"
                # Let's just pick the first valid one or the one with largest min dist
                # To save time, let's just pick any valid one initially, then refine
                # Or, we can use a heuristic: prefer smaller x? Larger x?
                # Smaller x -> larger weight -> larger gaps?
                # Let's try to maximize the minimum difference between sums.
                
                # Optimization: Only compute min dist for a few candidates
                # For now, let's just take the first valid one to be fast
                # This is a very greedy approach
                best_x = x
                best_sums = sums
                break # Take the first valid one
            
            if best_x is None:
                break
                
            S.append(best_x)
            remaining.remove(best_x)
            
        return S

    # Try to find a set of size 16, 17, etc.
    # We can use local search to improve the set
    
    best_set = greedy_construct()
    
    # If the greedy set is small, try random local search
    target_size = 16
    
    # Local search
    current_set = list(best_set)
    if len(current_set) < target_size:
        # Try to add elements
        pass
        
    # Let's use a randomized local search for better results
    # Start with a random set of size 16
    # Check if valid
    # If not, try to swap elements to fix collisions
    
    def random_local_search():
        # Start with a random set of size 16
        S = random.sample(range(1, 23), 16)
        
        # Check if valid
        if is_dissociated(S):
            return S
            
        # Local search
        for _ in range(10000):
            if time.time() > deadline:
                break
                
            # Find a collision
            sums = get_subset_sums(S)
            if len(sums) == (1 << len(S)):
                return S
                
            # Pick a random element in S and swap with a random element outside
            i = random.randint(0, len(S)-1)
            outside = list(set(range(1, 23)) - set(S))
            if not outside:
                break
            j = random.choice(outside)
            
            old_x = S[i]
            S[i] = j
            
            # Check if the new set is better (fewer collisions)
            # This is complex. Let's just accept if it's valid.
            # If not, we might need to revert.
            # For simplicity, let's just try random swaps and check validity.
            
            if is_dissociated(S):
                return S
                
            # Revert if not valid? Or keep going?
            # Let's revert to avoid getting stuck in invalid states
            S[i] = old_x
            
        return S

    # Try multiple random starts
    for _ in range(100):
        if time.time() > deadline:
            break
        S = random_local_search()
        if len(S) > len(best_set):
            best_set = S
            
    # Final check
    if not is_dissociated(best_set):
        # Fallback to a known good set if available
        # Or return the largest valid subset of best_set
        # For now, return best_set if it's valid, else a smaller one
        # We should ensure we return a valid set
        # Let's do a final greedy pass on best_set to ensure validity
        # Actually, if best_set is not valid, we need to remove elements until it is
        while best_set and not is_dissociated(best_set):
            # Remove a random element
            idx = random.randint(0, len(best_set)-1)
            best_set.pop(idx)
            
    return best_set
