"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

def _norm(t):
    return tuple(sorted(t))

# Precompute: for each triple t, the set of "conflict triples" - triples that
# together with t would be part of a K(2,2,2) completion
# Actually, let's think differently. A K(2,2,2) is 8 specific triples.
# When adding triple t, we check if there exist x',y',z' such that all 8 transversals are present.

# Precompute for each triple t: for each possible (xp,yp,zp) not using x,y,z,
# what are the other 7 transversals needed?
# This is too large to precompute fully. Instead, let's optimize the check.

def _is_valid_add(trips_set, t):
    """Check if adding triple t to trips_set creates a K(2,2,2)."""
    x, y, z = t
    used = {x, y, z}
    for xp in range(V):
        if xp in used:
            continue
        for yp in range(V):
            if yp in used or yp == xp:
                continue
            for zp in range(V):
                if zp in used or zp == xp or zp == yp:
                    continue
                # Check the 7 other transversal triples
                if (x, y, zp) not in trips_set:
                    continue
                if (x, yp, z) not in trips_set:
                    continue
                if (x, yp, zp) not in trips_set:
                    continue
                if (xp, y, z) not in trips_set:
                    continue
                if (xp, y, zp) not in trips_set:
                    continue
                if (xp, yp, z) not in trips_set:
                    continue
                if (xp, yp, zp) not in trips_set:
                    continue
                return False  # Would create K(2,2,2)
    return True

def _greedy(order, trips_set, result, deadline):
    """Continue greedy from current state."""
    for t in order:
        if time.time() >= deadline:
            break
        if t in trips_set:
            continue
        if _is_valid_add(trips_set, t):
            trips_set.add(t)
            result.append(list(t))

def _local_search(trips_set, result, deadline, max_iters=50):
    """Try to improve by removing one triple and adding others."""
    best_len = len(result)
    best_result = [list(t) for t in result]
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Try removing a random triple and re-adding some
        if not result:
            break
        
        idx = random.randint(0, len(result) - 1)
        removed = result[idx]
        removed_set = set(removed)
        
        # Temporarily remove
        trips_set.discard(tuple(removed))
        result.pop(idx)
        
        # Try to add back triples that were previously blocked
        # Get triples that contain any of the removed triple's vertices
        candidates = []
        for t in ALL_TRIPLES:
            if t in trips_set:
                continue
            # Check if this triple shares vertices with removed
            if any(v in removed_set for v in t):
                if _is_valid_add(trips_set, t):
                    candidates.append(t)
        
        # Add as many as possible
        random.shuffle(candidates)
        added = 0
        for t in candidates:
            if time.time() >= deadline:
                break
            if t in trips_set:
                continue
            if _is_valid_add(trips_set, t):
                trips_set.add(t)
                result.append(list(t))
                added += 1
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = [list(t) for t in result]
        
        # If we didn't improve, put the removed triple back
        if len(result) <= best_len and best_len > len(result):
            # Put it back
            if _is_valid_add(trips_set, tuple(removed)):
                trips_set.add(tuple(removed))
                result.append(removed)
    
    return best_result, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best = None
    best_len = -1
    
    # Base: all triples through vertex 0
    base = [tuple([0, a, b]) for a in range(1, V) for b in range(a + 1, V)]
    base_set = set(base)
    
    # Other triples
    other_triples = [t for t in ALL_TRIPLES if 0 not in t]
    
    # Strategy: multiple randomized greedy with different heuristics
    for attempt in range(300):
        if time.time() >= deadline:
            break
        
        # Different strategies for ordering
        strategy = attempt % 4
        
        if strategy == 0:
            # Random order
            shuffled_base = base[:]
            random.shuffle(shuffled_base)
            shuffled_other = other_triples[:]
            random.shuffle(shuffled_other)
            order = shuffled_base + shuffled_other
        elif strategy == 1:
            # Base first in random order, then others sorted by "degree" (fewer conflicts first)
            shuffled_base = base[:]
            random.shuffle(shuffled_base)
            # For others, sort randomly but with some structure
            shuffled_other = other_triples[:]
            random.shuffle(shuffled_other)
            order = shuffled_base + shuffled_other
        elif strategy == 2:
            # Interleave base and other
            shuffled_base = base[:]
            random.shuffle(shuffled_base)
            shuffled_other = other_triples[:]
            random.shuffle(shuffled_other)
            # Interleave
            order = []
            bi, oi = 0, 0
            while bi < len(shuffled_base) or oi < len(shuffled_other):
                if bi < len(shuffled_base) and (oi >= len(shuffled_other) or random.random() < 0.5):
                    order.append(shuffled_base[bi])
                    bi += 1
                else:
                    if oi < len(shuffled_other):
                        order.append(shuffled_other[oi])
                        oi += 1
        else:
            # Other triples first (random), then base
            shuffled_other = other_triples[:]
            random.shuffle(shuffled_other)
            shuffled_base = base[:]
            random.shuffle(shuffled_base)
            order = shuffled_other + shuffled_base
        
        trips_set = set()
        result = []
        
        _greedy(order, trips_set, result, deadline)
        
        # Local search improvement
        if time.time() < deadline - 10:
            result, best_len_candidate = _local_search(trips_set, result, deadline, max_iters=20)
            if best_len_candidate > best_len:
                best_len = best_len_candidate
                best = result
        else:
            if len(result) > best_len:
                best_len = len(result)
                best = [list(t) for t in result]
        
        # Early termination check
        if time.time() >= deadline - 5:
            break
    
    if best is None:
        return [list(t) for t in base]
    
    return best
