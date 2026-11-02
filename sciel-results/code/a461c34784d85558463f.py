"""Rewritten: MIS via branch-and-bound (Tomita) on the norm-4 candidate graph."""
import time
import numpy as np
from itertools import combinations

D = 11

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _generate_norm4_candidates():
    """All nonzero integer vectors with squared norm 4 in R^11."""
    vecs = []
    # Type A: single coordinate ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
    # Type B: two coordinates ±1
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def _greedy_color_bound(conflict, P, n):
    """Greedy coloring of the conflict subgraph induced by P.
    Returns (color_of, color_classes) where color_of[v] is the color index (0-based)
    and color_classes[c] is a list of vertices with color c.
    The number of colors is an upper bound on the size of any independent set in P.
    """
    color_of = {}
    color_classes = []
    uncolored = list(P)
    # Order by degree within P for better coloring (fewer colors)
    # But for speed with numpy, we'll use a simple sequential approach
    # Actually, let's use the standard Tomita ordering: sort by degree in the
    # conflict subgraph, highest first, to get a good upper bound.
    
    # Compute degrees within P for ordering
    P_set = set(P)
    degrees = {}
    for v in P:
        # Count neighbors of v that are in P
        deg = np.sum(conflict[v, :])  # total degree, but we need within P
        # More efficient: use the conflict matrix
        deg_in_P = 0
        for u in P:
            if u != v and conflict[v, u]:
                deg_in_P += 1
        degrees[v] = deg_in_P
    
    # Sort by degree descending
    uncolored.sort(key=lambda x: -degrees[x])
    
    while uncolored:
        c = len(color_classes)
        color_class = []
        remaining = []
        for v in uncolored:
            # Check if v conflicts with any vertex already in this color class
            can_add = True
            for u in color_class:
                if conflict[v, u]:
                    can_add = False
                    break
            if can_add:
                color_class.append(v)
                color_of[v] = c
            else:
                remaining.append(v)
        color_classes.append(color_class)
        uncolored = remaining
    
    return color_of, color_classes


def _bb_mcs(conflict, n, deadline, best_size_ref):
    """Tomita BB-MCS for Maximum Independent Set.
    conflict: n x n boolean numpy array.
    Returns the best independent set found.
    """
    best_set = []
    best_size = best_size_ref[0]
    best_set_ref = [None]

    # We use a recursive approach with the standard Tomita branching.
    # State: (P, X, C) where P are unprocessed candidates, X are excluded,
    # C are color classes. For simplicity and to match the original structure,
    # we'll use a slightly different but equivalent formulation.

    def _search(P, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                best_size_ref[0] = best_size
            return

        # Upper bound check
        if len(current_set) + len(P) <= best_size:
            return

        # Greedy coloring for upper bound
        color_of, color_classes = _greedy_color_bound(conflict, P, n)
        num_colors = len(color_classes)

        # Process color classes in reverse order (highest color first)
        for ci in range(num_colors - 1, -1, -1):
            if time.time() > deadline:
                return
            # Upper bound: current_set + (ci + 1)
            if len(current_set) + ci + 1 <= best_size:
                break

            color_class = color_classes[ci]
            for v in color_class:
                if v not in P:
                    continue
                # Branch: include v
                new_set = current_set + [v]
                # New P: vertices in P that are not v and not conflicting with v
                # And not already excluded
                new_P = []
                for u in P:
                    if u != v and not conflict[v, u]:
                        new_P.append(u)
                _search(new_P, new_set)
                if time.time() > deadline:
                    return
                # Remove v from P
                P.remove(v)

    P_initial = list(range(n))
    _search(P_initial, [])
    return best_set


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    candidates = _generate_norm4_candidates()
    n = len(candidates)

    # Build conflict matrix: conflict if dot product > 2
    dots = candidates @ candidates.T
    conflict = (dots > 2).astype(bool)
    np.fill_diagonal(conflict, False)

    # Greedy initialization to get a good lower bound
    rng = np.random.default_rng(42)
    best_size = 0
    best_set = []

    def _greedy_init(order):
        kept = []
        for idx in order:
            v = idx
            ok = True
            for u in kept:
                if conflict[v, u]:
                    ok = False
                    break
            if ok:
                kept.append(v)
        return kept

    # Multiple greedy restarts
    for _ in range(30):
        if time.time() > deadline:
            break
        order = rng.permutation(n)
        s = _greedy_init(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Degree-based greedy: sort by degree (ascending) for better packing
    degrees = np.sum(conflict, axis=1)
    order_deg = np.argsort(degrees)
    s = _greedy_init(order_deg)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Now run BB-MCS to try to improve
    best_size_ref = [best_size]
    remaining = deadline - time.time()
    if remaining > 1.0:
        result_set = _bb_mcs(conflict, n, deadline, best_size_ref)
        if result_set and len(result_set) > best_size:
            best_set = result_set
            best_size = len(result_set)

    if best_size == 0:
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    result = candidates[best_set]

    # Verify validity
    n_res = len(result)
    max_norm_sq = int(np.max(np.sum(result ** 2, axis=1)))
    valid = True
    for i in range(n_res):
        for j in range(i + 1, n_res):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break

    if not valid:
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    return result
