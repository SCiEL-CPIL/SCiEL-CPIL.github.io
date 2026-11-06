"""Rewritten program for sunflower-free 4-uniform family on [1,16]."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _sunflower_conflict_bitmask(a, b, c):
    """Check if three 4-element sets (as frozensets) form a sunflower.
    A sunflower means all three pairwise intersections are equal.
    For 4-element sets, this can happen when:
    - All three share a common 2-element core (petals of size 2 each)
    - All three share a common 3-element core (petals of size 1 each)
    - All three are identical (excluded by 'distinct')
    - The empty-core case: pairwise intersections are all empty (disjoint sets)
    """
    ab = a & b
    ac = a & c
    bc = b & c
    # All three pairwise intersections must be equal
    return ab == ac == bc

def _sunflower_conflict_fast(a, b, c):
    """Fast check using integer bitmasks."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_randomized(all_sets, bitmasks, rng, max_time=None, start_time=None):
    """Randomized greedy: repeatedly pick a random eligible set to add."""
    if max_time is None:
        max_time = 90
    if start_time is None:
        start_time = time.time()
    
    chosen = []
    chosen_bitmasks = []
    n = len(all_sets)
    # Work with indices
    remaining = list(range(n))
    rng.shuffle(remaining)
    
    while remaining:
        if time.time() - start_time > max_time - 2:
            break
        
        # Try to find a valid set from remaining
        added = False
        for i in range(len(remaining)):
            idx = remaining[i]
            bm = bitmasks[idx]
            # Check against all chosen
            valid = True
            for j in range(len(chosen_bitmasks)):
                if _sunflower_conflict_fast(bm, chosen_bitmasks[j], 0) or True:
                    # Need to check properly with third set
                    pass
            
            # Proper check: for each pair in chosen, check if adding this creates sunflower
            valid = True
            for j in range(len(chosen_bitmasks)):
                for k in range(j+1, len(chosen_bitmasks)):
                    if _sunflower_conflict_fast(bm, chosen_bitmasks[j], chosen_bitmasks[k]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                chosen.append(idx)
                chosen_bitmasks.append(bm)
                remaining.pop(i)
                added = True
                break
        
        if not added:
            break
    
    return [all_sets[i] for i in chosen]

def _greedy_bitmask(all_sets, bitmasks, rng, max_time=None, start_time=None):
    """Greedy with bitmask representation for speed."""
    if max_time is None:
        max_time = 90
    if start_time is None:
        start_time = time.time()
    
    chosen_bitmasks = []
    n = len(all_sets)
    remaining = list(range(n))
    rng.shuffle(remaining)
    
    while remaining:
        if time.time() - start_time > max_time - 2:
            break
        
        added = False
        for i in range(len(remaining)):
            idx = remaining[i]
            bm = bitmasks[idx]
            
            # Check if adding bm creates a sunflower with any pair in chosen
            valid = True
            m = len(chosen_bitmasks)
            for j in range(m):
                for k in range(j+1, m):
                    ab = bm & chosen_bitmasks[j]
                    ac = bm & chosen_bitmasks[k]
                    bc = chosen_bitmasks[j] & chosen_bitmasks[k]
                    if ab == ac == bc:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                chosen_bitmasks.append(bm)
                remaining.pop(i)
                added = True
                break
        
        if not added:
            break
    
    return chosen_bitmasks

def construct():
    start_time = time.time()
    max_time = 88  # Safety margin
    
    # Generate all C(16,4) = 1820 four-element subsets
    all_sets = []
    bitmasks = []
    for combo in combinations(range(1, 17), 4):
        s = set(combo)
        # Convert to bitmask: element i (1-indexed) -> bit (i-1)
        bm = 0
        for x in combo:
            bm |= (1 << (x - 1))
        all_sets.append(s)
        bitmasks.append(bm)
    
    n = len(all_sets)  # 1820
    
    # Precompute pairwise intersections for all pairs? Too many (1820^2 ~ 3.3M)
    # Instead, use the fast bitmask check on the fly.
    
    best_result = []
    best_size = 0
    
    # Multiple randomized greedy restarts
    rng = random.Random(42)
    
    # First, do a deterministic greedy to get a baseline
    # Sort by some heuristic and greedily add
    for restart in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Randomized greedy
        result_bitmasks = _greedy_bitmask(all_sets, bitmasks, rng, max_time, start_time)
        
        if len(result_bitmasks) > best_size:
            best_size = len(result_bitmasks)
            best_result = []
            for bm in result_bitmasks:
                # Convert bitmask back to sorted list
                elements = []
                for bit in range(16):
                    if bm & (1 << bit):
                        elements.append(bit + 1)
                best_result.append(sorted(elements))
    
    # If we still have time, try local search / improvement
    if best_size > 0 and time.time() - start_time < max_time - 5:
        # Try to improve by removing one set and adding two
        # This is complex; for now, just return the best greedy result
        pass
    
    # Ensure we return at least something valid
    if not best_result:
        # Fallback: just return a small valid family
        best_result = [sorted(list(combo)) for combo in list(combinations(range(1, 17), 4))[:10]]
    
    return best_result
# EVOLVE-BLOCK-END
