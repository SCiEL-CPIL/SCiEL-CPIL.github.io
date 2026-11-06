"""Construct a large graph with no regular induced subgraph on >=6 vertices."""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0   # safety margin before the 150s kill

def _regular_sets(n, A, min_size=6):
    """Return (sets, sizes): every vertex subset (size>=min_size) that is regular.
    Uses numpy matrix multiplication for speed."""
    sets = []
    sizes = []
    for size in range(min_size, n + 1):
        degs = A.sum(axis=1)
        # find candidate subsets of given size where all degrees equal
        # iterate over combinations
        from itertools import combinations
        for comb in combinations(range(n), size):
            d = degs[list(comb)]
            if d.min() == d.max():
                sets.append(list(comb))
                sizes.append(size)
    return sets, sizes

def _violation_count(n, A, min_size=6):
    """Count all regular induced subsets of size >= min_size."""
    total = 0
    from itertools import combinations
    degs = A.sum(axis=1)
    for size in range(min_size, n + 1):
        for comb in combinations(range(n), size):
            d = degs[list(comb)]
            if d.min() == d.max():
                total += 1
    return total

def _any_violation(n, A, min_size=6):
    """Return a violating subset (list) or None. Fast early exit."""
    from itertools import combinations
    degs = A.sum(axis=1)
    for size in range(min_size, n + 1):
        for comb in combinations(range(n), size):
            d = degs[list(comb)]
            if d.min() == d.max():
                return list(comb)
    return None

def _try_add_vertex(n, A, seed=0):
    """Try to add a new vertex (index n) connected to some subset of existing
    vertices so that no regular induced subgraph of size>=6 appears.
    Returns new A (n+1 x n+1) on success, else None."""
    rng = np.random.default_rng(seed)
    m = n * n
    # candidate connection patterns: random subsets of existing vertices
    # We'll try a batch of random patterns and check violations.
    for attempt in range(400):
        if time.time() > _DEADLINE - 3:
            return None
        # random pattern
        conn = (rng.random(n) < 0.5)
        # build new A
        A2 = np.zeros((n + 1, n + 1), dtype=np.int8)
        A2[:n, :n] = A
        A2[:n, n] = conn
        A2[n, :n] = conn
        # check violations only involving the new vertex (size>=6)
        # plus check that existing graph still has none (it doesn't, by invariant)
        # A regular subset of size>=6 either avoids new vertex (already none) or includes it.
        # Check subsets that include new vertex.
        from itertools import combinations
        found = False
        for size in range(6, n + 1):
            need = size - 1  # other vertices from existing
            if need > n:
                continue
            for comb in combinations(range(n), need):
                # degrees within this induced set (size 'size')
                # degree of new vertex = conn.sum() over comb
                dv_new = conn[list(comb)].sum()
                # degrees of existing vertices within comb (within the full set of size 'size')
                # degree of existing v = (A[comb].sum over comb) + conn[v]
                sub = A[np.ix_(list(comb), list(comb))]
                d_existing = sub.sum(axis=1) + conn[list(comb)]
                alldeg = np.concatenate([d_existing, [dv_new]])
                if alldeg.min() == alldeg.max():
                    found = True
                    break
            if found:
                break
        if not found:
            return A2
    return None

def construct():
    global _DEADLINE
    # Start from a valid base graph. Use a known small regular-free graph.
    # We'll build up greedily from n=6 with a random seed and grow.
    best_n = 5
    best_A = np.zeros((5, 5), dtype=np.int8)

    rng = np.random.default_rng(12345)

    # Try to start from a random valid 6-vertex graph, then grow.
    # First, find any valid graph of size 6 (no regular subset of size 6 means
    # the whole graph is not regular; smaller subsets <6 don't matter).
    # Actually for n=6, the only subset of size>=6 is the whole graph.
    # So we need a 6-vertex graph that is NOT regular. Easy.
    # But as we grow, we must ensure no regular subset of ANY size >=6.

    # Build up from n=6.
    # Start: find a valid 6-vertex graph (non-regular).
    A = None
    for seed in range(50):
        if time.time() > _DEADLINE - 5:
            break
        cand = (rng.random((6, 6)) < 0.5).astype(np.int8)
        np.fill_diagonal(cand, 0)
        # symmetrize
        cand = ((cand + cand.T) > 0).astype(np.int8)
        np.fill_diagonal(cand, 0)
        if _any_violation(6, cand) is None:
            A = cand
            break
    if A is None:
        # fallback: a clearly non-regular 6-graph
        A = np.zeros((6, 6), dtype=np.int8)
        # make vertex 0 connected to all, others form a path
        for i in range(1, 6):
            A[0, i] = 1
            A[i, 0] = 1
        A[1, 2] = A[2, 1] = 1
        A[2, 3] = A[3, 2] = 1
        A[3, 4] = A[4, 3] = 1
        A[4, 5] = A[5, 4] = 1
        # degrees: v0=5, v1=2, v2=2, v3=2, v4=2, v5=1 -> not regular, good
        if _any_violation(6, A) is not None:
            # shouldn't happen, but guard
            return {'n': 5, 'edges': []}

    n = 6
    if n > best_n:
        best_n = n
        best_A = A.copy()

    # Now try to grow from 6 upward.
    cur_A = A
    cur_n = 6
    seed_counter = 0
    while cur_n < 22:
        if time.time() > _DEADLINE - 10:
            break
        # try to add a vertex with multiple seeds
        new_A = None
        for seed in range(200):
            if time.time() > _DEADLINE - 5:
                break
            new_A = _try_add_vertex(cur_n, cur_A, seed=seed_counter + seed)
            if new_A is not None:
                break
            seed_counter += 1
        if new_A is None:
            # try a few more with different base perturbations? Keep simple: stop growing this branch.
            break
        cur_A = new_A
        cur_n += 1
        if cur_n > best_n:
            best_n = cur_n
            best_A = cur_A.copy()

    # Build edge list from best_A
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_A[i, j]:
                edges.append([i, j])

    # Final safety: verify no violation (with deadline guard).
    if time.time() < _DEADLINE - 1:
        v = _any_violation(best_n, best_A)
        if v is not None:
            # invalid; fall back to a smaller known-good by trimming? Return the last valid we trust.
            # We'll return the empty-ish safe graph to avoid a 0 score.
            # Actually best_A was verified at each step, so this shouldn't trigger.
            # As a guard, return n=5 empty.
            return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': edges}
