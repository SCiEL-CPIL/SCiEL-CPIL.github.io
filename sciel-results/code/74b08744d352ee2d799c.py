import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_EDGE_SET = set()
_ADJ = [set() for _ in range(_N)]
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))

# Bitmask adjacency for fast operations
_ADJ_MASK = [0] * _N
for u in range(_N):
    mask = 0
    for v in _ADJ[u]:
        mask |= (1 << v)
    _ADJ_MASK[u] = mask

# Edge -> index map and edge list
_EDGE_LIST = list(_EDGE_SET)
_EDGE_INDEX = {e: i for i, e in enumerate(_EDGE_LIST)}
_M = len(_EDGE_LIST)

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

def _biclique_edges(L, R):
    edges = set()
    for u in L:
        for v in R:
            edges.add((min(u, v), max(u, v)))
    return edges

def _grow_biclique(a, b, remaining_mask):
    """Grow a biclique from seed edge (a,b) using bitmask ops.
    remaining_mask: list of int bitmasks (per vertex) of remaining edges."""
    L = {a}
    R = {b}
    changed = True
    while changed:
        changed = False
        # mask of all R vertices
        mask_R = 0
        for v in R:
            mask_R |= (1 << v)
        # candidates for L: connected to all R, not in L or R
        for u in range(_N):
            if u in L or u in R:
                continue
            # u must be adjacent (in remaining) to every v in R
            if (remaining_mask[u] & mask_R) == mask_R:
                # also u must have at least one remaining edge to R (guaranteed if mask_R nonempty and equal)
                L.add(u)
                changed = True
        mask_L = 0
        for u in L:
            mask_L |= (1 << u)
        for v in range(_N):
            if v in L or v in R:
                continue
            if (remaining_mask[v] & mask_L) == mask_L:
                R.add(v)
                changed = True
    return list(L), list(R)

def _greedy_construct(deadline, rng=None):
    """Greedy set cover with bitmask biclique growing."""
    if rng is None:
        rng = random.Random()
    # remaining_mask: bitmask of remaining edges per vertex
    remaining_mask = [0] * _N
    for e in _EDGE_LIST:
        a, b = e
        remaining_mask[a] |= (1 << b)
        remaining_mask[b] |= (1 << a)

    bics = []
    edges_left = _M
    while edges_left > 0 and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0

        # Single-vertex stars
        for u in range(_N):
            m = remaining_mask[u]
            cnt = bin(m).count('1')
            if cnt > best_count:
                best_count = cnt
                R = []
                mm = m
                while mm:
                    lsb = mm & -mm
                    v = lsb.bit_length() - 1
                    R.append(v)
                    mm ^= lsb
                best_L = [u]
                best_R = R

        # Grow from random edges
        edges_list = list(_EDGE_LIST)
        rng.shuffle(edges_list)
        for e in edges_list:
            if time.time() >= deadline:
                break
            a, b = e
            # check edge still remaining
            if not (remaining_mask[a] & (1 << b)):
                continue
            L, R = _grow_biclique(a, b, remaining_mask)
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R

        if best_L is None or best_count == 0:
            # find any remaining edge
            for i in range(_N):
                if remaining_mask[i]:
                    lsb = remaining_mask[i] & -remaining_mask[i]
                    j = lsb.bit_length() - 1
                    best_L = [i]
                    best_R = [j]
                    break

        bics.append((list(best_L), list(best_R)))
        # remove edges
        mask_L = 0
        for u in best_L:
            mask_L |= (1 << u)
        mask_R = 0
        for v in best_R:
            mask_R |= (1 << v)
        for u in best_L:
            remaining_mask[u] &= ~mask_R
        for v in best_R:
            remaining_mask[v] &= ~mask_L
        edges_left -= best_count

    # any leftovers
    for i in range(_N):
        m = remaining_mask[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            bics.append(([i], [j]))
            remaining_mask[i] &= ~(1 << j)
            remaining_mask[j] &= ~(1 << i)
            m = remaining_mask[i]
    return bics

def _try_merge(bics, i, j):
    """Try to merge bicliques i and j. Return new list or None."""
    L1, R1 = bics[i][0], bics[i][1]
    L2, R2 = bics[j][0], bics[j][1]

    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    old_edges = e1 | e2

    candidates = []
    sL1, sR1, sL2, sR2 = set(L1), set(R1), set(L2), set(R2)

    # Full merges
    candidates.append((list(sL1 | sL2), list(sR1 | sR2)))
    candidates.append((list(sL1 | sR2), list(sR1 | sL2)))
    candidates.append((list(sR1 | sL2), list(sL1 | sR2)))
    candidates.append((list(sR1 | sR2), list(sL1 | sL2)))

    # Partial merges
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
        new_edges = _biclique_edges(L, R)
        if new_edges != old_edges:
            continue
        new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
        new_bics.append((L, R))
        return new_bics
    return None

def _local_search(bics, deadline):
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
            u, v = L1[0], R1[0]
            for j in range(n):
                if i == j:
                    continue
                L2, R2 = bics[j][0], bics[j][1]
                if u not in L2 and u not in R2:
                    if all((min(u, x), max(u, x)) in _EDGE_SET for x in R2):
                        new_edges = _biclique_edges(L2 + [u], R2)
                        old_edges = _biclique_edges(L2, R2)
                        expected = old_edges | {(min(u, v), max(u, v))}
                        if new_edges == expected:
                            bics[j] = (L2 + [u], R2)
                            bics.pop(i)
                            changed = True
                            break
                if not changed and v not in L2 and v not in R2:
                    if all((min(x, v), max(x, v)) in _EDGE_SET for x in L2):
                        new_edges = _biclique_edges(L2, R2 + [v])
                        old_edges = _biclique_edges(L2, R2)
                        expected = old_edges | {(min(u, v), max(u, v))}
                        if new_edges == expected:
                            bics[j] = (L2, R2 + [v])
                            bics.pop(i)
                            changed = True
                            break
            if changed:
                break
    return bics

def _enumerate_maximal_bicliques(seed_edges, deadline):
    """Enumerate inclusion-wise maximal bicliques from seed edges.
    Returns list of (L, R, edge_index_set)."""
    pool = {}  # frozenset(L) | frozenset(R) -> (L, R, edge_set)
    # also track by edge set for dedup
    seen_edge_sets = set()
    for a, b in seed_edges:
        if time.time() >= deadline:
            break
        L, R = _grow_biclique(a, b, _ADJ_MASK)
        # make it a valid biclique: ensure all cross edges are in graph
        # (growth already ensures this via adjacency)
        key = (frozenset(L), frozenset(R))
        if key in pool:
            continue
        es = _biclique_edges(L, R)
        # check all are real edges (they should be)
        if es <= _EDGE_SET:
            pool[key] = (list(L), list(R), es)
            seen_edge_sets.add(frozenset(es))
    return pool

def _branch_and_bound(candidates, deadline):
    """Branch-and-bound set cover: find minimum number of candidates covering all edges.
    candidates: list of (L, R, edge_index_set).
    Returns (count, selected_indices) or (None, None) if not solved in time."""
    # candidates as list of edge-index frozensets
    cand_edges = []
    for L, R, es in candidates:
        idx = frozenset(_EDGE_INDEX[e] for e in es)
        cand_edges.append(idx)

    # Also add single-edge bicliques as fallback candidates
    for e in _EDGE_LIST:
        idx = frozenset([_EDGE_INDEX[e]])
        cand_edges.append(idx)

    all_edges = frozenset(range(_M))
    best_count = len(cand_edges)  # upper bound: one per edge
    best_solution = None

    # Precompute: for each candidate, bitmask of edges it covers (as Python int)
    cand_mask = []
    for idx in cand_edges:
        m = 0
        for i in idx:
            m |= (1 << i)
        cand_mask.append(m)
    full_mask = (1 << _M) - 1

    # Sort candidates by size descending for better branching
    order = sorted(range(len(cand_mask)), key=lambda i: bin(cand_mask[i]).count('1'), reverse=True)
    # Deduplicate by mask
    seen_masks = set()
    dedup_order = []
    for i in order:
        if cand_mask[i] not in seen_masks:
            seen_masks.add(cand_mask[i])
            dedup_order.append(i)

    nodes = 0
    limit_nodes = 500000

    def bnb(remaining_mask, chosen, depth):
        nonlocal best_count, best_solution, nodes
        if time.time() >= deadline:
            return
        if remaining_mask == 0:
            if len(chosen) < best_count:
                best_count = len(chosen)
                best_solution = list(chosen)
            return
        nodes += 1
        if nodes > limit_nodes:
            return
        if len(chosen) >= best_count:
            return
        # lower bound: greedy packing
        # find a candidate that covers the most remaining edges
        # simple lower bound: remaining edges / max candidate size
        max_cand_size = 0
        for i in dedup_order:
            if cand_mask[i] & remaining_mask:
                s = bin(cand_mask[i] & remaining_mask).count('1')
                if s > max_cand_size:
                    max_cand_size = s
        if max_cand_size == 0:
            return
        lb = (bin(remaining_mask).count('1') + max_cand_size - 1) // max_cand_size
        if len(chosen) + lb >= best_count:
            return

        # pick the remaining edge with fewest candidate covers (most constrained)
        # find an edge in remaining_mask
        lsb = remaining_mask & -remaining_mask
        edge_idx = lsb.bit_length() - 1
        # candidates covering this edge
        covering = []
        for i in dedup_order:
            if cand_mask[i] & (1 << edge_idx):
                covering.append(i)
        if not covering:
            return
        # sort by coverage of remaining
        covering.sort(key=lambda i: bin(cand_mask[i] & remaining_mask).count('1'), reverse=True)

        for i in covering:
            chosen.append(i)
            bnb(remaining_mask & ~cand_mask[i], chosen, depth + 1)
            chosen.pop()

    bnb(full_mask, [], 0)
    if best_solution is not None:
        return best_count, best_solution
    return None, None

def _pool_set_cover(candidates, deadline):
    """Greedy set cover over candidate pool (fast, always valid)."""
    cand_edges = []
    for L, R, es in candidates:
        idx = frozenset(_EDGE_INDEX[e] for e in es)
        cand_edges.append(idx)
    for e in _EDGE_LIST:
        idx = frozenset([_EDGE_INDEX[e]])
        cand_edges.append(idx)

    remaining = set(range(_M))
    chosen = []
    while remaining:
        best_i = None
        best_cnt = 0
        for i, idx in enumerate(cand_edges):
            cnt = len(idx & remaining)
            if cnt > best_cnt:
                best_cnt = cnt
                best_i = i
        if best_i is None:
            break
        chosen.append(best_i)
        remaining -= cand_edges[best_i]
    return chosen

def construct():
    deadline = time.time() + 75.0
    best_partition = None
    best_count = float('inf')

    # Phase 1: Greedy + merge + absorb (multiple restarts)
    for seed in range(25):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed * 7919 + 13)
        part = _greedy_construct(deadline - 15, rng=rng)
        if not _valid_partition(part):
            continue
        part = _local_search(part, deadline - 10)
        part = _absorb_singletons(part, deadline - 8)
        if _valid_partition(part):
            cnt = len(part)
            if cnt < best_count:
                best_count = cnt
                best_partition = part
                if best_count <= 5:
                    break

    if best_partition is None:
        best_partition = _star_partition()
        best_count = len(best_partition)

    # Phase 2: Build candidate pool and do exact/BB polish
    if time.time() < deadline - 5:
        # Collect seed edges from current best partition + random edges
        seed_edges = set()
        for L, R in best_partition:
            for u in L:
                for v in R:
                    seed_edges.add((min(u, v), max(u, v)))
        # Add random edges for diversity
        rng2 = random.Random(42)
        for _ in range(200):
            e = rng2.choice(_EDGE_LIST)
            seed_edges.add(e)

        pool = _enumerate_maximal_bicliques(list(seed_edges), deadline - 5)
        # Add the bicliques from best_partition to the pool
        for L, R in best_partition:
            key = (frozenset(L), frozenset(R))
            if key not in pool:
                es = _biclique_edges(L, R)
                if es <= _EDGE_SET:
                    pool[key] = (list(L), list(R), es)

        candidates = list(pool.values())
        if candidates:
            # Try branch-and-bound
            bb_count, bb_sol = _branch_and_bound(candidates, deadline - 2)
            if bb_sol is not None and bb_count < best_count:
                # reconstruct partition
                part = []
                for idx in bb_sol:
                    L, R, es = candidates[idx]
                    part.append((list(L), list(R)))
                if _valid_partition(part):
                    best_count = bb_count
                    best_partition = part

            # Also try greedy set cover on pool as a fallback/improvement
            if time.time() < deadline - 1:
                gc_sol = _pool_set_cover(candidates, deadline - 1)
                if gc_sol:
                    part = []
                    for idx in gc_sol:
                        L, R, es = candidates[idx]
                        part.append((list(L), list(R)))
                    if _valid_partition(part) and len(part) < best_count:
                        best_count = len(part)
                        best_partition = part

    result = []
    for bic in best_partition:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])
    return result
