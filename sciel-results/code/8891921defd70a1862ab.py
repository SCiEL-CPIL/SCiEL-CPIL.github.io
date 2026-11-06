"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))

def _check_k222(trips_set, t, x, y, z):
    """Check if adding triple t={x,y,z} would create a K(2,2,2).
    We need to find x',y',z' such that all 8 transversal triples are in trips_set ∪ {t}.
    """
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
                # Check all 8 transversal triples
                # t itself is being added, so we check the other 7
                if (xp, y, z) not in trips_set and (xp, y, z) != t:
                    continue
                if (xp, y, zp) not in trips_set:
                    continue
                if (xp, yp, z) not in trips_set:
                    continue
                if (xp, yp, zp) not in trips_set:
                    continue
                if (x, yp, z) not in trips_set:
                    continue
                if (x, yp, zp) not in trips_set:
                    continue
                if (x, y, zp) not in trips_set:
                    continue
                return True
    return False

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
                # Check all 8 transversal triples (t is being added, so check other 7)
                # Normalize tuples for lookup
                def norm(triple):
                    return tuple(sorted(triple))
                
                # The 8 transversal triples:
                # (x,y,z), (x,y,zp), (x,yp,z), (x,yp,zp), (xp,y,z), (xp,y,zp), (xp,yp,z), (xp,yp,zp)
                # t = (x,y,z) is being added, so we need the other 7 to be in trips_set
                candidates = [
                    (x, y, zp),
                    (x, yp, z),
                    (x, yp, zp),
                    (xp, y, z),
                    (xp, y, zp),
                    (xp, yp, z),
                    (xp, yp, zp),
                ]
                all_present = True
                for c in candidates:
                    if norm(c) not in trips_set:
                        all_present = False
                        break
                if all_present:
                    return False  # Would create K(2,2,2)
    return True

def _greedy(order, deadline):
    """Greedily add triples in given order, skipping those that create K(2,2,2)."""
    trips_set = set()
    result = []
    for t in order:
        if time.time() >= deadline:
            break
        if t in trips_set:
            continue
        if _is_valid_add(trips_set, t):
            trips_set.add(t)
            result.append(list(t))
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best = None
    best_len = -1
    
    # Start with the base: all triples through vertex 0
    base = [tuple([0, a, b]) for a in range(1, V) for b in range(a + 1, V)]
    base_set = set(base)
    
    # Other triples to consider
    other_triples = [t for t in ALL_TRIPLES if 0 not in t]
    
    # Try multiple random orders
    for attempt in range(200):
        if time.time() >= deadline:
            break
        
        # Create a random order: base triples first (in random order), then others
        shuffled_base = base[:]
        random.shuffle(shuffled_base)
        shuffled_other = other_triples[:]
        random.shuffle(shuffled_other)
        
        order = shuffled_base + shuffled_other
        
        result = _greedy(order, deadline)
        
        if len(result) > best_len:
            best_len = len(result)
            best = result
            # Early termination if we've used all remaining time budget
            if time.time() >= deadline - 5:
                break
    
    if best is None:
        # Fallback: just return the base
        return [list(t) for t in base]
    
    return best
