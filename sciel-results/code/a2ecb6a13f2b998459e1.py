import sys
from itertools import combinations
from math import gcd
from functools import reduce

def construct():
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 23))
    
    # Integer values v_i = L // i
    # v_i are the scaled reciprocals
    vals = [L // i for i in range(1, 23)]
    
    # We want to find a subset of size K such that all subset sums of vals are distinct.
    # This is equivalent to: no non-empty subset sums to 0 when considering signs?
    # No, distinct subset sums means:
    # For any two distinct subsets A, B, sum(A) != sum(B).
    # This is equivalent to: for any non-empty subset C, sum(C) != 0? No.
    # It is equivalent to: for any disjoint non-empty subsets A, B, sum(A) != sum(B).
    # Which is equivalent to: for any subset D, sum(D) != 0 is NOT sufficient.
    # Example: 1, -1. Subsets: {}, {1}, {-1}, {1,-1}. Sums: 0, 1, -1, 0.
    # 0 appears twice. So {1, -1} is NOT dissociated in this sense.
    # But 1, 2. Sums: 0, 1, 2, 3. All distinct.
    # The condition is that the map $S \to \sum_{i \in S} v_i$ is injective.
    # This implies that no non-trivial linear combination $\sum c_i v_i = 0$ with $c_i \in \{-1, 0, 1\}$ exists.
    
    def is_dissociated(subset_indices):
        """
        Check if the subset of indices has distinct subset sums.
        Uses integer arithmetic.
        """
        sums = {0}
        for idx in subset_indices:
            val = vals[idx - 1]
            new_sums = {s + val for s in sums}
            # If any new sum is already in sums, collision
            if not sums.isdisjoint(new_sums):
                return False
            sums = sums.union(new_sums)
        return True

    def search_max_size(target_size):
        """
        Find a valid subset of size target_size using backtracking with integer sums.
        """
        # Heuristic: Sort indices by value?
        # Let's try to include indices that correspond to "large" values first?
        # Or just natural order?
        # Natural order 1..22 corresponds to values L/1, L/2, ... L/22.
        # L/1 is largest. L/22 is smallest.
        # Including large values first might prune faster?
        # Or including small values first?
        # In the previous float code, it used natural order 1..22.
        
        # Let's try a specific ordering that favors "independent" structures.
        # Powers of 2: 1, 2, 4, 8, 16.
        # Indices: 0, 1, 3, 7, 15 (0-indexed).
        # Let's try to prioritize these.
        
        # However, for a general DFS, natural order is often fine if pruning is good.
        # Let's stick to natural order for simplicity, but use integers.
        
        def dfs(start, current_subset, current_sums):
            if len(current_subset) == target_size:
                return current_subset
            
            # Pruning: not enough elements left
            if len(current_subset) + (22 - start + 1) < target_size:
                return None
            
            for i in range(start, 23):
                val = vals[i - 1]
                new_sums = {s + val for s in current_sums}
                
                if current_sums.isdisjoint(new_sums):
                    new_subset = current_subset + [i]
                    result = dfs(i + 1, new_subset, current_sums.union(new_sums))
                    if result is not None:
                        return result
            return None

        return dfs(1, [], {0})

    # Try size 11 first
    candidate = search_max_size(11)
    if candidate:
        return candidate
    
    # If 11 fails, try 10
    candidate = search_max_size(10)
    if candidate:
        return candidate
    
    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
