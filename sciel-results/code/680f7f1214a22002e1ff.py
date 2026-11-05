import sys
from itertools import combinations

def construct():
    # Known valid set of size 9
    fallback_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Reciprocals
    recips = [1.0 / i for i in range(1, 23)]
    
    def is_dissociated(subset_indices):
        """
        Check if the subset of indices (values 1..22) has distinct subset sums of reciprocals.
        """
        sums = {0.0}
        for idx in subset_indices:
            val = recips[idx - 1]
            new_sums = {s + val for s in sums}
            # Check for intersection
            if len(sums.union(new_sums)) != len(sums) + len(new_sums):
                return False
            sums = sums.union(new_sums)
        return True

    def search_max_size(target_size):
        """
        Find a valid subset of size target_size using backtracking.
        Returns the subset (list of ints) or None.
        """
        # We want to find ANY valid set of this size.
        # We'll use a recursive helper that tries to pick elements.
        # To prune effectively, we should check dissociation early.
        
        # Optimization: Pre-check. If we pick an element, we must ensure the current set is dissociated.
        
        def dfs(start, current_subset, current_sums):
            if len(current_subset) == target_size:
                return current_subset
            
            # Pruning: if we don't have enough numbers left to reach target_size
            if len(current_subset) + (22 - start + 1) < target_size:
                return None
            
            # Try including numbers from start to 22
            for i in range(start, 23):
                # Include i
                new_val = recips[i - 1]
                # Calculate new sums
                new_sums = {s + new_val for s in current_sums}
                
                # Check if any collision occurred
                # Collision if intersection of current_sums and new_sums is not empty
                # Actually, we just need to check if new_sums has duplicates within itself or with current_sums?
                # No, the property is that ALL 2^k sums are distinct.
                # The new sums are formed by adding new_val to all previous sums.
                # A collision occurs if:
                # 1. Two new sums are equal: s1 + v == s2 + v => s1 == s2 (already distinct)
                # 2. A new sum equals an old sum: s + v == t (where s, t in current_sums)
                # So we just need to check if (current_sums + new_val) intersects current_sums.
                
                if current_sums.isdisjoint(new_sums):
                    # No collision
                    new_subset = current_subset + [i]
                    result = dfs(i + 1, new_subset, current_sums.union(new_sums))
                    if result is not None:
                        return result
                # If collision, we skip this i (cannot include it in a valid set with this current_subset)
                
            return None

        # Start DFS
        # We can start with empty set
        # To speed up, we might want to try including small numbers or large numbers?
        # Usually, including 1 is good because it shifts the range.
        # Let's just run the DFS.
        
        # Note: The order of iteration matters for finding a solution quickly.
        # Iterating 1..22 is standard.
        
        return dfs(1, [], {0.0})

    # Try to find a set of size 10
    # If found, return it.
    # We add a timeout mechanism? The DFS should be fast due to pruning.
    # If it takes too long, we might need to bail.
    # Given the small size, it should be fine.
    
    candidate = search_max_size(10)
    if candidate:
        return candidate
    
    # If size 10 not found quickly or doesn't exist, return the known size 9
    # We can also try size 9 with a different set if we want, but the seed is fine.
    return fallback_set
