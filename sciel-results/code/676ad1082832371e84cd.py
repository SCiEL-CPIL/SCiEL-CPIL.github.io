import time
import random

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']

# Pre-compute host adjacency as bitmasks (static)
HOST_ADJ = [0] * N
for _u, _v in P['edges']:
    HOST_ADJ[_u] |= (1 << _v)
    HOST_ADJ[_v] |= (1 << _u)
HOST_EDGE_SET = set(frozenset((u, v)) for u, v in P['edges'])

def _bits_to_list(mask):
    res = []
    m = mask
    while m:
        lsb = m & -m
        res.append(lsb.bit_length() - 1)
        m ^= lsb
    return res

def _is_valid_partition(bicliques):
    """Verify exact partition: disjoint sides, all cross pairs host edges,
    cross-edge sets partition all host edges exactly once."""
    if not bicliques:
        return False
    seen = set()
    for b in bicliques:
        L, R = b
        if not L or not R:
            return False
        for x in L:
            if x in seen:
                return False
        for x in R:
            if x in seen:
                return False
        seen.update(L)
        seen.update(R)
        for u in L:
            for v in R:
                if (u, v) in seen or (v, u) in seen:
                    return False
                if (u, v) not in HOST_EDGE_SET and (v, u) not in HOST_EDGE_SET:
                    return False
                seen.add((u, v))
    if len(seen) != 2 * len(HOST_EDGE_SET):
        return False
    return True

def _try_merge(bicliques, deadline):
    """Iteratively merge compatible bicliques to reduce the count."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                return bicliques
            for j in range(i + 1, n):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                L1s, R1s, L2s, R2s = set(L1), set(R1), set(L2), set(R2)
                # disjointness
                if (L1s & L2s) or (R1s & R2s) or (L1s & R2s) or (L2s & R1s):
                    continue
                # all cross edges between the merged sides must be host edges
                ok = True
                for u in L1s:
                    for v in R2s:
                        if (u, v) not in HOST_EDGE_SET and (v, u) not in HOST_EDGE_SET:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    for u in L2s:
                        for v in R1s:
                            if (u, v) not in HOST_EDGE_SET and (v, u) not in HOST_EDGE_SET:
                                ok = False
                                break
                            if not ok:
                                break
                if not ok:
                    continue
                # merge
                newL = list(L1s | L2s)
                newR = list(R1s | R2s)
                bicliques.pop(j)
                bicliques.pop(i)
                bicliques.append([newL, newR])
                changed = True
                break
            if changed:
                break
    return bicliques

def _local_search(bicliques, deadline):
    """Try to reduce count by dropping a biclique and re-covering its edges
    using existing bicliques (absorbing) or a fresh star."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = [b[:] for b in bicliques]
        n = len(cur)
        for k in range(n):
            if time.time() > deadline:
                return bicliques
            L, R = cur[k]
            # edges covered by this biclique
            edges = set()
            for u in L:
                for v in R:
                    edges.add(frozenset((u, v)))
            # try to re-cover each edge using other bicliques
            other = [b for i, b in enumerate(cur) if i != k]
            # map each edge to a covering biclique index (in 'other')
            assign = {}
            for e in edges:
                a, b = list(e)
                placed = False
                for idx, (Lb, Rb) in enumerate(other):
                    Lbs, Rbs = set(Lb), set(Rb)
                    if a in Lbs and b in Rbs:
                        assign[e] = idx
                        placed = True
                        break
                    if b in Lbs and a in Rbs:
                        assign[e] = idx
                        placed = True
                        break
                if not placed:
                    break
            if len(assign) != len(edges):
                continue
            # build new bicliques from 'other' with absorbed vertices
            new_bics = []
            for idx, (Lb, Rb) in enumerate(other):
                newL = set(Lb)
                newR = set(Rb)
                for e, ai in assign.items():
                    if ai == idx:
                        a, b = list(e)
                        if a in newL and b in newR:
                            continue
                        # add the vertex that isn't already on the right/left appropriately
                        # determine orientation: which endpoint is in newL vs newR
                        if a in newL:
                            newR.add(b)
                        elif b in newL:
                            newR.add(a)
                        elif a in newR:
                            newL.add(b)
                        elif b in newR:
                            newL.add(a)
                        else:
                            # neither present: put a on left, b on right
                            newL.add(a)
                            newR.add(b)
                if newL and newR:
                    new_bics.append([list(newL), list(newR)])
            if len(new_bics) < len(cur) and _is_valid_partition(new_bics):
                bicliques = new_bics
                improved = True
                break
    return bicliques

def _greedy_once(deadline, seed):
    random.seed(seed)
    adj = HOST_ADJ[:]
    bicliques = []
    while True:
        if time.time() > deadline:
            return None
        is_empty = True
        for u in range(N):
            if adj[u]:
                is_empty = False
                break
        if is_empty:
            break
        best_val = -1
        candidates = []
        # 1. Single vertices (Stars)
        for u in range(N):
            if adj[u]:
                r_mask = adj[u]
                val = r_mask.bit_count()
                if val > 0:
                    candidates.append((val, [u], r_mask))
        # 2. Pairs
        for u in range(N):
            if not adj[u]:
                continue
            for v in range(u + 1, N):
                if not adj[v]:
                    continue
                r_mask = adj[u] & adj[v]
                if r_mask:
                    val = 2 * r_mask.bit_count()
                    candidates.append((val, [u, v], r_mask))
        # 3. Triples
        for u in range(N):
            if not adj[u]:
                continue
            for v in range(u + 1, N):
                if not adj[v]:
                    continue
                common_uv = adj[u] & adj[v]
                if not common_uv:
                    continue
                if 3 * common_uv.bit_count() <= best_val:
                    continue
                for w in range(v + 1, N):
                    if not adj[w]:
                        continue
                    r_mask = common_uv & adj[w]
                    if r_mask:
                        val = 3 * r_mask.bit_count()
                        candidates.append((val, [u, v, w], r_mask))
        if not candidates:
            return None
        max_val = max(c[0] for c in candidates)
        top_candidates = [c for c in candidates if c[0] >= max_val * 0.9]
        if not top_candidates:
            top_candidates = [max(candidates, key=lambda x: x[0])]
        val, L, R_mask = random.choice(top_candidates)
        bicliques.append([L, _bits_to_list(R_mask)])
        for u in L:
            adj[u] &= ~R_mask
            m = R_mask
            while m:
                lsb = m & -m
                r_idx = lsb.bit_length() - 1
                adj[r_idx] &= ~(1 << u)
                m ^= lsb
    if any(adj):
        return None
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before the 90s kill

    best_bicliques = None
    best_count = float('inf')

    # Phase 1: randomized greedy restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
        bicliques = _greedy_once(deadline, restart)
        if bicliques is None:
            continue
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques

    # Fallback to stars if nothing found
    if best_bicliques is None:
        adj = HOST_ADJ[:]
        bicliques = []
        for u in range(N):
            if adj[u]:
                bicliques.append([[u], _bits_to_list(adj[u])])
                m = adj[u]
                while m:
                    lsb = m & -m
                    r_idx = lsb.bit_length() - 1
                    adj[r_idx] &= ~(1 << u)
                    adj[u] &= ~lsb
                    m ^= lsb
        best_bicliques = bicliques
        best_count = len(bicliques)

    # Phase 2: polish the best solution with merging + local search
    polished = _try_merge(best_bicliques, deadline)
    polished = _local_search(polished, deadline)
    if _is_valid_partition(polished) and len(polished) < best_count:
        best_bicliques = polished
        best_count = len(polished)

    # Final type conversion (exact Python ints, lists)
    final_bicliques = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        final_bicliques.append([left, right])

    return final_bicliques

# EVOLVE-BLOCK-END
