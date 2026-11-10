"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n = 256
    
    def is_sidon(s):
        """Check if set s is Sidon (all pair sums including a+a are unique)."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def get_diffs(s):
        """Get all nonzero differences from set s."""
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d = s[j] - s[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def can_add_to_sidon(s, x):
        """Check if x can be added to Sidon set s while maintaining Sidon property."""
        # Check x+x against existing sums
        xx = x + x
        for a in s:
            if a + a == xx:
                return False
        # Check x+a against existing sums
        for i in range(len(s)):
            for j in range(i, len(s)):
                if s[i] + s[j] == x + s[i] or s[i] + s[j] == x + s[j]:
                    return False
        # More efficient: check all pairs involving x
        for a in s:
            if (x + a) in set(s[i] + s[j] for i in range(len(s)) for j in range(i, len(s))):
                return False
        # Check x+x
        if xx in set(s[i] + s[i] for i in range(len(s))):
            return False
        return True
    
    def greedy_construct(order, max_size=64):
        """Greedy construction using given element order."""
        sets = [[], []]
        diff_sets = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
            
            for side in (0, 1):
                if len(sets[side]) >= max_size:
                    continue
                
                # Check Sidon property
                candidate = sets[side] + [x]
                if not is_sidon(candidate):
                    continue
                
                # Check disjoint differences
                new_diffs = get_diffs(candidate)
                old_diffs = get_diffs(sets[side])
                added_diffs = new_diffs - old_diffs
                
                if added_diffs & diff_sets[1 - side]:
                    continue
                
                # Also need to check that existing diffs of this side don't conflict
                # with the other side's diffs (already ensured by invariant)
                
                sets[side] = candidate
                diff_sets[side] = new_diffs
                break
        
        return sets
    
    def local_improve(sets, order):
        """Try to add more elements by removing some and re-adding."""
        sets = [list(s) for s in sets]
        diff_sets = [get_diffs(s) for s in sets]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in order:
                if x in sets[0] or x in sets[1]:
                    continue
                for side in (0, 1):
                    if len(sets[side]) >= 64:
                        continue
                    candidate = sets[side] + [x]
                    if not is_sidon(candidate):
                        continue
                    new_diffs = get_diffs(candidate)
                    added_diffs = new_diffs - diff_sets[side]
                    if added_diffs & diff_sets[1 - side]:
                        continue
                    sets[side] = candidate
                    diff_sets[side] = new_diffs
                    improved = True
                    break
        return sets
    
    # Strategy: Try multiple random orderings and greedy constructions
    best_sets = [[], []]
    best_score = 0
    
    # First, try a deterministic smaller-first ordering
    base_order = list(range(1, n + 1))
    sets = greedy_construct(base_order)
    sets = local_improve(sets, base_order)
    score = len(sets[0]) + len(sets[1])
    if score > best_score:
        best_score = score
        best_sets = [list(s) for s in sets]
    
    # Try reverse ordering
    rev_order = list(range(n, 0, -1))
    sets = greedy_construct(rev_order)
    sets = local_improve(sets, rev_order)
    score = len(sets[0]) + len(sets[1])
    if score > best_score:
        best_score = score
        best_sets = [list(s) for s in sets]
    
    # Try random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        sets = greedy_construct(order)
        sets = local_improve(sets, order)
        score = len(sets[0]) + len(sets[1])
        if score > best_score:
            best_score = score
            best_sets = [list(s) for s in sets]
    
    # Final improvement pass on the best sets
    final_order = list(range(1, n + 1))
    best_sets = local_improve(best_sets, final_order)
    
    # Verify validity
    for s in best_sets:
        assert is_sidon(s), "Sidon property violated"
    
    d0 = get_diffs(best_sets[0])
    d1 = get_diffs(best_sets[1])
    assert not (d0 & d1), "Difference sets not disjoint"
    
    return best_sets

# EVOLVE-BLOCK-END
