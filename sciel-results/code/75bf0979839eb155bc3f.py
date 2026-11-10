import time
import sys

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums 
    of different cardinalities are disjoint.
    
    Strategy: Backtracking search with descending heuristic.
    We attempt to build the set by adding the largest available numbers first.
    This tends to separate the subset sum layers (S_k) more effectively,
    reducing collisions between different cardinalities.
    """
    
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    # T[k] is a set of integers representing sums of subsets of size k
    # Start with empty set: T[0] = {0}
    T = [set() for _ in range(61)]
    T[0] = {0}
    current_set = []
    used = set()
    
    best_size = 0
    best_set = []
    
    # Candidate numbers in descending order
    # We only care about numbers 1..256
    candidates = list(range(256, 0, -1))
    
    def update_best():
        nonlocal best_size, best_set
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    def is_valid_addition(x):
        """
        Check if adding x to current_set maintains the constraint.
        Constraint: S'_k ∩ S'_j = empty for k != j.
        Since T_k ∩ T_j is empty for k != j (invariant), we only need to check
        if new sums {s + x | s in T[k-1]} intersect with any existing T[j] (j != k).
        Also, we must ensure new sums for different k don't intersect?
        If s+x = t+x, then s=t. If s in T[k-1] and t in T[j-1], and s=t,
        then T[k-1] intersects T[j-1]. If k!=j, then k-1 != j-1.
        Since T layers are disjoint, T[k-1] ∩ T[j-1] is empty.
        So new sums for different k are automatically disjoint.
        We only check against existing T[j].
        """
        max_k = len(current_set) + 1
        for k in range(1, max_k + 1):
            if not T[k-1]:
                continue
            new_sums = {s + x for s in T[k-1]}
            # Check against all existing T[j] where j != k
            # Note: T[j] for j > len(current_set) is empty.
            for j in range(0, len(current_set) + 1):
                if j == k:
                    continue
                if T[j] and (new_sums & T[j]):
                    return False
        return True

    def backtrack(idx):
        nonlocal T, current_set, used, best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        # Update best if we have a larger set
        update_best()
        
        # If we have reached max possible size, stop
        if len(current_set) == 60:
            return
            
        # Try adding candidates from idx onwards
        # Optimization: If we can't add enough more to beat best_size, prune
        # Max remaining elements = 256 - idx (roughly)
        # But we are limited by 60 total.
        
        for i in range(idx, len(candidates)):
            if time.time() > deadline:
                return
                
            x = candidates[i]
            if x in used:
                continue
                
            if is_valid_addition(x):
                # Add x
                current_set.append(x)
                used.add(x)
                
                # Update T
                old_T = [s.copy() for s in T]
                new_max_k = len(current_set)
                for k in range(1, new_max_k + 1):
                    for s in old_T[k-1]:
                        T[k].add(s + x)
                        
                # Recurse
                backtrack(i + 1)
                
                # Backtrack
                # Restore T
                T = old_T
                current_set.pop()
                used.remove(x)
                
                # Optional: If we found a solution of size 60, we can stop
                if best_size == 60:
                    return

    # Start backtracking
    # We start with idx=0 (smallest number)
    backtrack(0)
    
    # If backtracking didn't find anything better than initial (which is 0), 
    # but we might have a valid set from the greedy-like path if we didn't backtrack enough?
    # No, backtrack explores.
    
    # Ensure we return a valid list. If best_set is empty, maybe we should fallback to a known valid small set?
    # The empty set is valid (size 0). But we want to maximize.
    # If best_size is 0, return empty list? Or try a simple greedy to ensure at least some size?
    # The problem implies we should return a list of ints.
    
    # If best_set is empty, let's try a quick forward greedy to get a baseline, 
    # though backtracking should have found at least 1 element (1 is always valid).
    if not best_set:
        # Fallback: simple forward greedy for safety
        T_fallback = [set() for _ in range(61)]
        T_fallback[0] = {0}
        current_set_fallback = []
        used_fallback = set()
        
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in used_fallback:
                continue
            valid = True
            max_k = len(current_set_fallback) + 1
            for k in range(1, max_k + 1):
                if not T_fallback[k-1]:
                    continue
                new_sums = {s + x for s in T_fallback[k-1]}
                for j in range(0, len(current_set_fallback) + 1):
                    if j == k:
                        continue
                    if T_fallback[j] and (new_sums & T_fallback[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_set_fallback.append(x)
                used_fallback.add(x)
                old_T = [s.copy() for s in T_fallback]
                for k in range(1, len(current_set_fallback) + 1):
                    for s in old_T[k-1]:
                        T_fallback[k].add(s + x)
                if len(current_set_fallback) == 60:
                    break
        best_set = current_set_fallback

    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
