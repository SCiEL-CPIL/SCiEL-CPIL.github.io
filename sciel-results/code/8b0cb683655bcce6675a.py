"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_ALL = len(ALL_TRIPLES)  # 560

# Build index mapping for fast lookup
_TRIPLE_TO_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}

def _triple_in_set(trips_set, a, b, c):
    """Check if triple {a,b,c} is in trips_set (as a frozenset or sorted tuple)."""
    t = (min(a,b,c), min(max(a,b,c), max(a,b,c)), max(a,b,c))
    # Actually let's just use a set of frozensets or sorted tuples
    return t in trips_set

def _check_k222(trips_set, t):
    """
    Check if adding triple t would create a K(2,2,2).
    t = (x, y, z) with x < y < z.
    We need to find x', y', z' distinct from {x,y,z} and from each other
    such that all 8 transversal triples of pairs (x,x'),(y,y'),(z,z') are present.
    The 8 triples are all combinations choosing one from each pair.
    t is one of them. The other 7 must be in trips_set.
    """
    x, y, z = t
    used = {x, y, z}
    remaining = [p for p in range(V) if p not in used]
    
    # For each choice of x', y', z' from remaining (distinct),
    # check if the other 7 transversal triples are in trips_set.
    
    n = len(remaining)
    for i in range(n):
        xp = remaining[i]
        for j in range(i+1, n):
            yp = remaining[j]
            for k in range(j+1, n):
                zp = remaining[k]
                # Check the 7 other transversal triples
                # They are:
                # (x, y, zp), (x, yp, z), (x, yp, zp),
                # (xp, y, z), (xp, y, zp), (xp, yp, z), (xp, yp, zp)
                if not _triple_in_set(trips_set, x, y, zp):
                    continue
                if not _triple_in_set(trips_set, x, yp, z):
                    continue
                if not _triple_in_set(trips_set, x, yp, zp):
                    continue
                if not _triple_in_set(trips_set, xp, y, z):
                    continue
                if not _triple_in_set(trips_set, xp, y, zp):
                    continue
                if not _triple_in_set(trips_set, xp, yp, z):
                    continue
                if not _triple_in_set(trips_set, xp, yp, zp):
                    continue
                return True
    return False

def _greedy(restart_seed, deadline, use_base=False):
    """Greedy algorithm with random ordering."""
    random.seed(restart_seed)
    trips_set = set()
    triples_list = []
    
    if use_base:
        # Start with all triples through point 0
        for a in range(1, V):
            for b in range(a + 1, V):
                t = (0, a, b)
                trips_set.add(t)
                triples_list.append(t)
    
    remaining = [t for t in ALL_TRIPLES if t not in trips_set]
    random.shuffle(remaining)
    
    for t in remaining:
        if time.time() > deadline:
            break
        if not _check_k222(trips_set, t):
            trips_set.add(t)
            triples_list.append(t)
    
    return triples_list

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = None
    best_count = 0
    
    # Multiple restarts with different strategies
    for restart in range(30):
        if time.time() > deadline:
            break
        use_base = (restart % 2 == 0)
        result = _greedy(restart, deadline, use_base=use_base)
        if len(result) > best_count:
            best_count = len(result)
            best_result = [list(t) for t in result]
    
    if best_result is None:
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
