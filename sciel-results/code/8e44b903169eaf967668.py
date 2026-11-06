import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
N_WORDS = (N_TRIPLES + 63) // 64  # 9

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

def _has_presence(presence, idx):
    """Check if triple idx is in the presence bitmask."""
    return (presence[idx >> 6] >> (idx & 63)) & 1

def _check_conflict(a, b, c, presence):
    """Check if adding triple (a,b,c) creates a K(2,2,2).
    
    K(2,2,2): 6 distinct vertices a,a',b,b',c,c' with all 8 transversal triples.
    We fix a,b,c as one corner and look for a',b',c' such that all 8 exist.
    """
    for a_p in range(V):
        if a_p == a or a_p == b or a_p == c:
            continue
        # Easy check 1: (a_p, b, c) must be present
        if not _has_presence(presence, _tidx(a_p, b, c)):
            continue
        # Easy check 2: (a, b_p, c) for some b_p... we need to find b_p, c_p
        # Find valid b_p: distinct from a,b,c,a_p; (a,b_p,c) and (a_p,b_p,c) present
        valid_bp = []
        for b_p in range(V):
            if b_p == a or b_p == b or b_p == c or b_p == a_p:
                continue
            if not _has_presence(presence, _tidx(a, b_p, c)):
                continue
            if not _has_presence(presence, _tidx(a_p, b_p, c)):
                continue
            valid_bp.append(b_p)
        if not valid_bp:
            continue
        # Find valid c_p: distinct from a,b,c,a_p; (a,b,c_p) and (a_p,b,c_p) present
        valid_cp = []
        for c_p in range(V):
            if c_p == a or c_p == b or c_p == c or c_p == a_p:
                continue
            if not _has_presence(presence, _tidx(a, b, c_p)):
                continue
            if not _has_presence(presence, _tidx(a_p, b, c_p)):
                continue
            valid_cp.append(c_p)
        if not valid_cp:
            continue
        # Check pairs (b_p, c_p) with b_p != c_p
        for b_p in valid_bp:
            for c_p in valid_cp:
                if b_p == c_p:
                    continue
                if not _has_presence(presence, _tidx(a, b_p, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a_p, b_p, c_p)):
                    continue
                return True
    return False

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts. Returns (count, presence)."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict(a, b, c, presence):
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def construct():
    start = time.time()
    deadline = start + 115  # safety margin for 150s wall-clock limit
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    # Multiple restarts with different random orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        if count > best_count:
            best_count = count
            best_presence = presence[:]
    
    # Extract triples
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
    return result

# Pre-seed random for reproducibility across runs is NOT desired; we want variety.
# But we do want a reasonable seed for the first call.
random.seed(42)
