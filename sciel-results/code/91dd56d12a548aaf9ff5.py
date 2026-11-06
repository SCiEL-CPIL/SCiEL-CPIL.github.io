"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated(sums, v):
    """Check if adding v keeps subset sums distinct.
    Equivalent to checking if sums and sums+v are disjoint."""
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Map a in 1..22 to v = L // a
    # We want to find a subset of indices (a's) such that their corresponding v's form a dissociated set.
    
    # Heuristic ordering:
    # Larger v's (smaller a's) are more restrictive.
    # Let's try ordering by v descending (a ascending) first, as this often prunes the search space faster for max cardinality?
    # Or ascending v?
    # Let's try a few orderings.
    
    # Ordering 1: a = 1, 2, ..., 22 (v = L, L/2, ..., L/22) -> Descending v
    # Ordering 2: a = 22, 21, ..., 1 (v = L/22, ..., L) -> Ascending v
    # Ordering 3: Randomized or specific heuristic?
    
    # Let's stick to two strong orderings to keep it fast.
    orderings = [
        [a for a in range(1, 23)],      # Descending v
        [a for a in range(22, 0, -1)]   # Ascending v
    ]
    
    best_subset = []
    best_size = 0
    
    for ordering in orderings:
        if time.time() > deadline:
            break
            
        vals = [_L // a for a in ordering]
        
        # Backtracking search
        # State: idx, current_sums (frozenset for efficiency? set is mutable, frozenset is hashable but we don't need to hash state, just check membership)
        # Using set is fine.
        
        def backtrack(idx, current_sums, current_set):
            nonlocal best_subset, best_size
            
            if time.time() > deadline:
                return
                
            # Update best if current is better
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_subset = current_set.copy()
                
            if idx == len(ordering):
                return
                
            a = ordering[idx]
            v = vals[idx]
            
            # Try including a
            if _is_dissociated(current_sums, v):
                # Create new sums
                # Optimization: if len(current_sums) is small, set union is fast.
                # new_sums = current_sums | {s + v for s in current_sums}
                # We can just pass a new set.
                new_sums = current_sums | {s + v for s in current_sums}
                backtrack(idx + 1, new_sums, current_set + [a])
            
            # Try excluding a
            backtrack(idx + 1, current_sums, current_set)
        
        backtrack(0, {0}, [])
        
    if not best_subset:
        # Fallback: A known small dissociated set
        # Powers of 2 are dissociated: 1, 2, 4, 8, 16
        # Can we add more?
        # 1, 2, 4, 8, 16, 32? 32 is not in [1,22].
        # So [1, 2, 4, 8, 16] is a valid set of size 5.
        # Let's try to return something valid immediately if search fails completely.
        return [1, 2, 4, 8, 16]
        
    return sorted(best_subset)
# EVOLVE-BLOCK-END
