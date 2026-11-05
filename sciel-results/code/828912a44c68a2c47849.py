"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def is_valid(S, target=200):
        """Check if no subset of S sums to target using bitset DP."""
        if not S:
            return True
        bits = 1
        for x in S:
            if x <= target:
                bits |= (bits << x)
        return (bits >> target) & 1 == 0

    def find_bad_subset(S, target=200):
        """Find a subset of S that sums to target. Returns list of elements or None."""
        # Simple DP to track parent
        # dp[s] = set of elements used to reach sum s? 
        # Too slow. 
        # Use bitset to check existence, then backtrack if needed.
        # For now, just return None if valid, else we need a specific element to remove.
        # We can just pick a random element from S and try removing it.
        pass

    # Known valid set: {66, ..., 100}, size 35
    best_S = set(range(66, 101))
    best_size = len(best_S)
    
    # Strategy 1: Local search starting from {66..100}
    # Try to add elements from 1..65 by removing conflicts
    # This is complex. 
    # Alternative: Start with larger sets and prune.
    
    def prune_to_valid(S, target=200):
        """
        Greedily remove elements from S until it is valid.
        Returns the valid subset.
        """
        S = set(S)
        while not is_valid(S, target):
            # Find a subset summing to target
            # We need to find WHICH elements form the sum.
            # Use DP with path reconstruction
            dp = [None] * (target + 1)
            dp[0] = []
            for x in S:
                if x > target:
                    continue
                # Iterate backwards to avoid using x multiple times
                for s in range(target, x - 1, -1):
                    if dp[s] is None and dp[s - x] is not None:
                        dp[s] = dp[s - x] + [x]
            if dp[target] is not None:
                bad_subset = dp[target]
                # Remove one element from bad_subset
                # Heuristic: remove the element that appears in the most "potential" bad subsets?
                # Or just random.
                # Let's try removing the largest element to keep the set "dense" at the bottom?
                # Or smallest?
                # Removing a large element might break fewer large sums.
                # Let's try random selection from the bad subset.
                x_to_remove = random.choice(bad_subset)
                S.remove(x_to_remove)
            else:
                # Should not happen if is_valid says False
                break
        return S

    # Try to start with sets larger than 35 and prune
    # {61..100} size 40
    # {60..100} size 41
    # {55..100} size 46
    
    candidates_starts = [
        set(range(61, 101)),
        set(range(60, 101)),
        set(range(55, 101)),
        set(range(50, 101))
    ]
    
    for start_set in candidates_starts:
        if time.time() > deadline:
            break
        # Run multiple random pruning passes
        for _ in range(50):
            if time.time() > deadline:
                break
            random.seed() # Reset seed? No, just use current state
            current_S = prune_to_valid(start_set)
            if len(current_S) > best_size:
                best_S = current_S
                best_size = len(current_S)
                # If we found a better one, we might try to improve it further
                # But prune_to_valid is greedy and might get stuck.
                
    # Local search on best_S
    # Try swapping elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        outside = set(range(1, 101)) - best_S
        if not outside:
            break
        # Try removing one from best_S and adding two from outside
        for remove in list(best_S):
            if time.time() > deadline:
                break
            S_without = best_S - {remove}
            # Try pairs from outside
            outside_list = list(outside)
            random.shuffle(outside_list)
            for i, add1 in enumerate(outside_list):
                if time.time() > deadline:
                    break
                for add2 in outside_list[i+1:]:
                    new_S = S_without | {add1, add2}
                    if len(new_S) > len(best_S):
                        if is_valid(new_S):
                            best_S = new_S
                            best_size = len(best_S)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    # Final check
    if not is_valid(best_S):
        # Fallback to known valid
        best_S = set(range(66, 101))
        
    result = list(best_S)
    assert len(result) > 0
    assert is_valid(set(result))
    return result
# EVOLVE-BLOCK-END
