"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))

def _check_k222(current_set, t):
    """Check if adding triple t would create a K(2,2,2)."""
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
                if ((xp, y, z) in current_set and
                    (x, yp, z) in current_set and
                    (x, y, zp) in current_set and
                    (xp, yp, z) in current_set and
                    (xp, y, zp) in current_set and
                    (x, yp, zp) in current_set and
                    (xp, yp, zp) in current_set):
                    return True
    return False

def _greedy_random():
    """
    Run one greedy pass with a random ordering of triples.
    """
    triples = ALL_TRIPLES[:]
    random.shuffle(triples)
    current_set = set()
    
    for t in triples:
        if not _check_k222(current_set, t):
            current_set.add(t)
    
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_set = set()
    best_size = 0
    
    # Initial run
    s = _greedy_random()
    best_set = s
    best_size = len(s)
    
    # Loop until deadline
    while time.time() < deadline:
        s = _greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Convert set of tuples to list of lists for output
    return [list(t) for t in best_set]
