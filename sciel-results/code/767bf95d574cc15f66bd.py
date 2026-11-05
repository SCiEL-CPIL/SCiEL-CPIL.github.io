import time
import random

N = 16
DEADLINE = 145.0  # safety margin before 150s kill

# All 4-subsets of range(16)
_SUBSETS4 = None
# Perfect matchings: list of 24 tuples of (row, col) pairs
_PERM_MATCHINGS = None

def _init():
    global _SUBSETS4, _PERM_MATCHINGS
    if _SUBSETS4 is None:
        _SUBSETS4 = [tuple(s) for s in __import__('itertools').combinations(range(N), 4)]
        # all 24 perfect matchings of 4 rows to 4 cols
        import itertools
        _PERM_MATCHINGS = [tuple(zip(range(4), p)) for p in itertools.permutations(range(4))]

def _has_perfect_matching(rows, cols, adj):
    """rows, cols are 4-tuples; adj is a dict/set of (r,c) present.
       Returns True if there is a perfect matching using only present edges."""
    # build 4x4 bitmask of which (row_i, col_j) edges exist
    for perm in _PERM_MATCHINGS:
        ok = True
        for r, c in perm:
            if (rows[r], cols[c]) not in adj:
                ok = False
                break
        if ok:
            return True
    return False

def _check_bad_pair(a_set, b_set, adj):
    """Return True if the induced bipartite subgraph on a_set (rows) and b_set (cols)
       contains Q3 as a subgraph."""
    rows = a_set
    cols = b_set
    # row sums and col sums must each be >= 3
    for r in rows:
        s = 0
        for c in cols:
            if (r, c) in adj:
                s += 1
        if s < 3:
            return False
    for c in cols:
        s = 0
        for r in rows:
            if (r, c) in adj:
                s += 1
        if s < 3:
            return False
    # perfect matching check
    return _has_perfect_matching(rows, cols, adj)

def _creates_q3(u, v, adj, side_a):
    """Check if adding edge (u,v) creates a Q3.
       side_a: set of vertices considered 'row' side; the rest are 'col' side.
       We only need to check 4-sets containing u on row side and 4-sets containing v on col side,
       OR u on col side and v on row side."""
    # u must be in a row 4-set, v in a col 4-set (or swapped)
    # Case 1: u in rows, v in cols
    for a_set in _SUBSETS4:
        if u not in a_set:
            continue
        # b_set must contain v and be disjoint from a_set
        for b_set in _SUBSETS4:
            if v not in b_set:
                continue
            if any(x in a_set for x in b_set):
                continue
            if _check_bad_pair(a_set, b_set, adj):
                return True
    # Case 2: u in cols, v in rows
    for b_set in _SUBSETS4:
        if u not in b_set:
            continue
        for a_set in _SUBSETS4:
            if v not in a_set:
                continue
            if any(x in b_set for x in a_set):
                continue
            if _check_bad_pair(a_set, b_set, adj):
                return True
    return False

def _is_q3_free(adj, side_a):
    """Full check: any bad pair?"""
    for a_set in _SUBSETS4:
        for b_set in _SUBSETS4:
            if any(x in a_set for x in b_set):
                continue
            if _check_bad_pair(a_set, b_set, adj):
                return False
    return True

def _try_build(seed):
    rng = random.Random(seed)
    # random bipartition: choose 8 vertices as side A
    verts = list(range(N))
    rng.shuffle(verts)
    side_a = frozenset(verts[:8])
    side_b = frozenset(verts[8:])
    # all candidate edges between side_a and side_b
    edges = [(a, b) for a in side_a for b in side_b]
    rng.shuffle(edges)
    adj = set()
    for (u, v) in edges:
        if time.time() > DEADLINE:
            break
        if _creates_q3(u, v, adj, side_a):
            continue
        adj.add((u, v))
    # verify
    if not _is_q3_free(adj, side_a):
        # fallback: remove edges until free (shouldn't happen)
        return None
    return [[u, v] for (u, v) in adj]

def construct():
    _init()
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]  # K3,13, 39 edges
    best_count = 39
    seed = 0
    while time.time() < DEADLINE:
        result = _try_build(seed)
        if result is not None:
            cnt = len(result)
            if cnt > best_count:
                best_count = cnt
                best_edges = result
        seed += 1
    return best_edges
