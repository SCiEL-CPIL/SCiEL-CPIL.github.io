import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_EDGE_SET = set()
_ADJ = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))

# Precompute edge lookup as a set of frozensets for faster membership
_EDGE_FS = {frozenset((min(a,b), max(a,b))) for a,b in P['edges']}

def _is_edge(u, v):
    return frozenset((u, v)) in _EDGE_FS


def _star_partition():
    res = []
    for u in range(_N):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res


def _valid_partition(bics):
    covered = set()
    for bic in bics:
        L, R = bic[0], bic[1]
        if not L or not R:
            return False
        if len(L) != len(set(L)) or len(R) != len(set(R)):
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == _EDGE_SET


def _grow_biclique(a, b, remaining):
    """Grow a biclique from seed edge (a,b) by expanding both sides."""
    # Start with L={a}, R={b}
    L = {a}
    R = {b}
    changed = True
    while changed:
        changed = False
        # Try to add vertices to L that connect to all of R
        for u in list(_ADJ[b]):
            if u in L:
                continue
            if all(_is_edge(u, v) and (min(u,v), max(u,v)) in remaining for v in R):
                L.add(u)
                changed = True
        # Try to add vertices to R that connect to all of L
        for v in list(_ADJ[a]):
            if v in R:
                continue
            if all(_is_edge(u, v) and (min(u,v), max(u,v)) in remaining for u in L):
                R.add(v)
                changed = True
    return list(L), list(R)


def _greedy_construct(deadline, rng=None):
    """Grow-based greedy set cover with optional randomization for diversification."""
    if rng is None:
        rng = random.Random()
    remaining = set(_EDGE_SET)
    bics = []
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0

        # Single-vertex left sides (stars)
        for u in range(_N):
            R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in remaining]
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R

        # Seed-edge bicliques: grow from each remaining edge
        edges_list = list(remaining)
        if rng:
            rng.shuffle(edges_list)
        for e in edges_list:
            if time.time() >= deadline:
                break
            a, b = e
            L, R = _grow_biclique(a, b, remaining)
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R

        if best_L is None or best_count == 0:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
            best_count = 1

        bics.append((list(best_L), list(best_R)))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))

    for e in list(remaining):
        bics.append(([e[0]], [e[1]]))
    return bics


def _try_merge(bics, i, j):
    """Try to merge bicliques i and j. Return new list or None."""
    L1, R1 = bics[i][0], bics[i][1]
    L2, R2 = bics[j][0], bics[j][1]

    # Compute edge sets for the two bicliques
    e1 = set()
    for u in L1:
        for v in R1:
            e1.add((min(u, v), max(u, v)))
    e2 = set()
    for u in L2:
        for v in R2:
            e2.add((min(u, v), max(u, v)))
    old_edges = e1 | e2

    # Try multiple orientations including partial merges
    candidates = []
    sL1, sR1, sL2, sR2 = set(L1), set(R1), set(L2), set(R2)

    # Full merges (4 orientations)
    candidates.append((list(sL1 | sL2), list(sR1 | sR2)))
    candidates.append((list(sL1 | sR2), list(sR1 | sL2)))
    candidates.append((list(sR1 | sL2), list(sL1 | sR2)))
    candidates.append((list(sR1 | sR2), list(sL1 | sL2)))

    # Partial merges: subset of L1 with all of L2, etc.
    # Try: (L1 ∩ something) - actually let's try keeping one side fixed
    # Orientation: keep R1, merge L1 with L2
    candidates.append((list(sL1 | sL2), list(sR1)))
    candidates.append((list(sL1), list(sR1 | sR2)))
    candidates.append((list(sL2 | sR1), list(sR2 | sL1)))
    candidates.append((list(sR2 | sL1), list(sL2 | sR1)))

    seen = set()
    for L, R in candidates:
        if not L or not R:
            continue
        if set(L) & set(R):
            continue
        key = (frozenset(L), frozenset(R))
        if key in seen:
            continue
        seen.add(key)
        # Check all cross pairs are edges
        valid = True
        for u in L:
            for v in R:
                if (min(u, v), max(u, v)) not in _EDGE_SET:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        # Check no extra edges beyond old_edges
        new_edges = set()
        for u in L:
            for v in R:
                new_edges.add((min(u, v), max(u, v)))
        if new_edges != old_edges:
            continue
        # Merge successful
        new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
        new_bics.append((L, R))
        return new_bics
    return None


def _local_search(bics, deadline):
    """Repeatedly merge any pair of bicliques until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics


def _absorb_singletons(bics, deadline):
    """Try to absorb single-edge bicliques into larger ones."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            L1, R1 = bics[i][0], bics[i][1]
            if len(L1) * len(R1) != 1:
                continue
            # This is a single-edge biclique
            u, v = L1[0], R1[0]
            for j in range(n):
                if i == j:
                    continue
                L2, R2 = bics[j][0], bics[j][1]
                # Try adding u to L2 if u connects to all of R2
                if u not in L2 and u not in R2:
                    if all(_is_edge(u, x) for x in R2):
                        # Check this doesn't create extra edges
                        new_edges = set()
                        for a in L2 + [u]:
                            for b in R2:
                                new_edges.add((min(a,b), max(a,b)))
                        old_edges = set()
                        for a in L2:
                            for b in R2:
                                old_edges.add((min(a,b), max(a,b)))
                        # The new biclique should cover old_edges + {(u,v)}
                        expected = old_edges | {(min(u,v), max(u,v))}
                        if new_edges == expected:
                            bics[j] = (L2 + [u], R2)
                            bics.pop(i)
                            changed = True
                            break
                # Try adding v to R2 if v connects to all of L2
                if not changed and v not in L2 and v not in R2:
                    if all(_is_edge(x, v) for x in L2):
                        new_edges = set()
                        for a in L2:
                            for b in R2 + [v]:
                                new_edges.add((min(a,b), max(a,b)))
                        old_edges = set()
                        for a in L2:
                            for b in R2:
                                old_edges.add((min(a,b), max(a,b)))
                        expected = old_edges | {(min(u,v), max(u,v))}
                        if new_edges == expected:
                            bics[j] = (L2, R2 + [v])
                            bics.pop(i)
                            changed = True
                            break
            if changed:
                break
    return bics


def construct():
    deadline = time.time() + 80.0

    best_partition = None
    best_count = float('inf')

    # Use diversified restarts with different random seeds
    for seed in range(40):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed * 7919 + 13)
        part = _greedy_construct(deadline - 3, rng=rng)
        if not _valid_partition(part):
            continue
        part = _local_search(part, deadline - 2)
        part = _absorb_singletons(part, deadline - 1)
        if _valid_partition(part):
            cnt = len(part)
            if cnt < best_count:
                best_count = cnt
                best_partition = part
                if best_count <= 5:
                    break

    if best_partition is None:
        best_partition = _star_partition()

    result = []
    for bic in best_partition:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    return result
