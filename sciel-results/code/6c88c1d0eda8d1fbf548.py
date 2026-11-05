import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Elements are 1 to 24
    # We want to maximize cardinality.
    # Strategy: Backtracking search.
    # Try to build the set by adding elements.
    # Order: Descending (24 to 1) seems to yield valid sets of large numbers easily.
    
    best_set = []
    
    # We will use a recursive function
    # current_set: list of selected elements
    # sums: set of Fraction sums of all subsets of current_set
    
    def is_valid_extension(current_set, sums, x):
        """
        Check if x can be added to current_set.
        Returns True if valid.
        """
        # 1. Check if 1/x is a sum of a subset of current_set
        target_x = Fraction(1, x)
        if target_x in sums:
            return False
            
        # 2. Check if adding x causes any existing u to become a subsum
        # For each u in current_set:
        # Is 1/u a sum of a subset of (current_set U {x}) \ {u}?
        # This is equivalent to: Is (1/u - 1/x) a sum of a subset of (current_set \ {u})?
        # Note: 1/u - 1/x must be positive for this to be a sum of positive reciprocals.
        # If 1/u <= 1/x, then 1/u - 1/x <= 0, so it cannot be a sum of positive terms (unless empty, but sum is 0, and 1/u > 0).
        # So we only need to check if 1/u > 1/x.
        
        for u in current_set:
            if u >= x: # 1/u <= 1/x
                continue
            # 1/u > 1/x
            diff = Fraction(1, u) - Fraction(1, x)
            # We need to check if diff is in sums(current_set \ {u})
            # Compute sums of current_set \ {u}
            # Optimization: If we did this often, we could cache.
            # But current_set changes.
            
            # Compute sums of current_set without u
            # We can do this by iterating through current_set and skipping u
            # This is O(2^|S|)
            # Since |S| is small, this is okay.
            
            # However, doing this for EVERY u in EVERY node is slow.
            # Let's optimize:
            # We can precompute the sums for the current state?
            # No, the state changes.
            
            # Let's just compute it.
            # To speed up, we can break early if we find the sum.
            
            # Generate all subset sums of current_set \ {u}
            # We can use a helper function
            if not _check_sum_in_set_without_u(current_set, u, diff):
                return False
                
        return True

    def _check_sum_in_set_without_u(current_set, u_excluded, target):
        """
        Check if target is a sum of a subset of current_set \ {u_excluded}.
        """
        # Get elements
        elems = [e for e in current_set if e != u_excluded]
        
        # If target is 0, empty set sums to 0. But target > 0.
        # Use DFS to find if target can be formed.
        # Or just generate all sums.
        
        # Since elems is small, generate all sums.
        # Optimization: Sort elems?
        
        sums = {Fraction(0)}
        for e in elems:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, e))
            sums.update(new_sums)
            
        return target in sums

    def dfs(idx, current_set, sums):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return
            
        # Update best
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Try to add more elements
        # We process elements in descending order: 24, 23, ...
        # idx is the current element to consider?
        # Let's just iterate from 24 down to 1.
        # But we need to avoid re-adding.
        # Let's pass a list of available elements.
        
        # Better: Pass a bitmask or just a list of remaining candidates.
        # Since we go 24->1, we can just pass the next index.
        
        for x in range(24, 0, -1):
            if x in current_set:
                continue
                
            # Check if we can add x
            if is_valid_extension(current_set, sums, x):
                # Compute new sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + Fraction(1, x))
                
                dfs(x, current_set + [x], new_sums)
                
                if time.time() > deadline:
                    return

    # Initial call
    # To help the search, we can start with a known good set?
    # Or just start empty.
    # Starting empty might explore too many branches.
    # Let's try to seed with the seed solution?
    # Seed: 13..24.
    
    # Let's try a greedy initialization first to get a good best_set quickly.
    # Greedy: Add elements 24 down to 1 if valid.
    greedy_set = []
    greedy_sums = {Fraction(0)}
    for x in range(24, 0, -1):
        if is_valid_extension(greedy_set, greedy_sums, x):
            greedy_set.append(x)
            new_sums = set(greedy_sums)
            for s in greedy_sums:
                new_sums.add(s + Fraction(1, x))
            greedy_sums = new_sums
            
    best_set = greedy_set[:]
    
    # Now do DFS to find better
    # We can start DFS from the greedy set?
    # Or restart DFS from empty but with the knowledge that we want large sets.
    # DFS from empty might be slow.
    # Let's do DFS from empty, but prune branches that can't beat best_set.
    # Upper bound: 24.
    
    # To make DFS faster, we can sort candidates?
    # The loop `for x in range(24, 0, -1)` tries large numbers first.
    # This is good because large numbers are "safer".
    
    # We need to be careful with the `dfs` signature.
    # The `idx` parameter in my draft was unused.
    # I will use a list of available elements.
    
    def dfs_search(current_set, sums, available):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if not available:
            return
            
        # Pruning: If len(current_set) + len(available) <= len(best_set), return
        if len(current_set) + len(available) <= len(best_set):
            return
            
        for i, x in enumerate(available):
            # Try adding x
            if is_valid_extension(current_set, sums, x):
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + Fraction(1, x))
                
                new_available = available[i+1:] # Only consider elements smaller than x?
                # Wait, if we process in descending order, and we pick x, 
                # the remaining candidates should be those < x to maintain order and avoid permutations.
                # But we want to maximize cardinality, not just find *a* set.
                # If we enforce order, we might miss sets where a smaller number is added before a larger one?
                # No, the set is unordered. 
                # If we iterate x from 24 down to 1, and pick x, 
                # the next candidates must be < x to avoid duplicate permutations of the same set.
                # This is standard for combinatorial search.
                
                dfs_search(current_set + [x], new_sums, [e for e in available if e < x])
                
                if time.time() > deadline:
                    return

    # Initial available: 24 down to 1
    dfs_search([], {Fraction(0)}, list(range(24, 0, -1)))
    
    return best_set

# EVOLVE-BLOCK-START
def construct():
    # Re-define to avoid global state issues if called multiple times?
    # The task says "Each evaluation must finish within 60s".
    # It implies construct() is called once per evaluation.
    
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    best_set = []
    
    # Helper functions inside construct to capture deadline
    
    def _check_sum_in_set_without_u(current_set, u_excluded, target):
        elems = [e for e in current_set if e != u_excluded]
        if not elems:
            return target == 0
        sums = {Fraction(0)}
        for e in elems:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, e))
            sums.update(new_sums)
        return target in sums

    def is_valid_extension(current_set, sums, x):
        target_x = Fraction(1, x)
        if target_x in sums:
            return False
        for u in current_set:
            if u >= x:
                continue
            diff = Fraction(1, u) - Fraction(1, x)
            if not _check_sum_in_set_without_u(current_set, u, diff):
                return False
        return True

    # Greedy init
    greedy_set = []
    greedy_sums = {Fraction(0)}
    for x in range(24, 0, -1):
        if is_valid_extension(greedy_set, greedy_sums, x):
            greedy_set.append(x)
            new_sums = set(greedy_sums)
            for s in greedy_sums:
                new_sums.add(s + Fraction(1, x))
            greedy_sums = new_sums
            
    best_set = greedy_set[:]
    
    def dfs_search(current_set, sums, available):
        nonlocal best_set, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if not available:
            return
            
        if len(current_set) + len(available) <= len(best_set):
            return
            
        for i, x in enumerate(available):
            if is_valid_extension(current_set, sums, x):
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + Fraction(1, x))
                
                # Next candidates must be < x
                new_available = [e for e in available[i+1:] if e < x]
                
                dfs_search(current_set + [x], new_sums, new_available)
                
                if time.time() > deadline:
                    return

    dfs_search([], {Fraction(0)}, list(range(24, 0, -1)))
    
    return best_set
# EVOLVE-BLOCK-END
