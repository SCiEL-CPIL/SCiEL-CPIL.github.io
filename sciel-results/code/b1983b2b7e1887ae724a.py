import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin for 90s limit

    n = P['n']
    edges = P['edges']
    m = len(edges)

    # Build adjacency bitmask and edge set
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v) if u < v else (v, u))

    def popcount(x):
        return bin(x).count('1')

    def mask_to_list(mask):
        result = []
        mm = mask
        while mm:
            lsb = mm & -mm
            result.append(lsb.bit_length() - 1)
            mm ^= lsb
        return result

    def list_to_mask(lst):
        mk = 0
        for x in lst:
            mk |= (1 << x)
        return mk

    def cross_edges(L_mask, R_mask):
        """Return the set of normalized edges between L and R (only host edges)."""
        out = set()
        ml = L_mask
        while ml:
            lsb = ml & -ml
            u = lsb.bit_length() - 1
            ml ^= lsb
            mr = R_mask
            while mr:
                lsb2 = mr & -mr
                v = lsb2.bit_length() - 1
                mr ^= lsb2
                if adj[u] & (1 << v):
                    out.add((u, v) if u < v else (v, u))
        return out

    def expand_biclique(u, v, rem_adj):
        """Expand biclique starting with (u,v) to maximal size within rem_adj."""
        L_mask = 1 << u
        R_mask = 1 << v
        full_mask = (1 << n) - 1

        changed = True
        while changed:
            changed = False
            # Expand L: candidates connected to all in R
            common_L = full_mask
            mr = R_mask
            while mr:
                lsb = mr & -mr
                y = lsb.bit_length() - 1
                mr ^= lsb
                common_L &= rem_adj[y]
            new_L = common_L & ~L_mask & ~R_mask
            if new_L:
                L_mask |= new_L
                changed = True
            # Expand R: candidates connected to all in L
            common_R = full_mask
            ml = L_mask
            while ml:
                lsb = ml & -ml
                x = lsb.bit_length() - 1
                ml ^= lsb
                common_R &= rem_adj[x]
            new_R = common_R & ~R_mask & ~L_mask
            if new_R:
                R_mask |= new_R
                changed = True
        return L_mask, R_mask

    def verify(bicliques):
        """Check the partition is valid: every cross pair is a host edge and
        cross-edge sets partition all edges exactly once. Returns True/False."""
        covered = []
        for b in bicliques:
            L = set(b[0])
            R = set(b[1])
            if not L or not R:
                return False
            if L & R:
                return False
            for u in L:
                for v in R:
                    if not (adj[u] & (1 << v)):
                        return False
                    covered.append((u, v) if u < v else (v, u))
        if len(covered) != len(set(covered)):
            return False
        return set(covered) == edge_set

    def greedy_partition(seed, max_candidates=6):
        """Greedy with multiple candidate evaluation per step."""
        random.seed(seed)
        rem_adj = adj[:]
        bicliques = []
        edges_left = m

        # degree ordering for bias
        degrees = [popcount(adj[u]) for u in range(n)]
        vert_order = sorted(range(n), key=lambda u: -degrees[u])

        while edges_left > 0:
            if time.time() > deadline:
                return None
            active_vertices = [u for u in range(n) if rem_adj[u] != 0]
            if not active_vertices:
                break

            candidates = []
            # deterministic high-degree candidates
            for u in vert_order:
                if rem_adj[u] == 0:
                    continue
                mm = rem_adj[u]
                vs = mask_to_list(mm)
                vs.sort(key=lambda v: -degrees[v])
                for v in vs[:4]:
                    candidates.append((u, v))
            # random candidates for diversity
            for _ in range(min(20, len(active_vertices) * 3)):
                u = random.choice(active_vertices)
                mask = rem_adj[u]
                if mask == 0:
                    continue
                count = popcount(mask)
                idx = random.randint(0, count - 1)
                v = None
                temp = mask
                for i in range(count):
                    lsb = temp & -temp
                    if i == idx:
                        v = lsb.bit_length() - 1
                        break
                    temp ^= lsb
                if v is not None:
                    candidates.append((u, v))

            if not candidates:
                break

            best_L = None
            best_R = None
            best_score = -1
            best_edges = 0

            for u, v in candidates[:max_candidates]:
                if not (rem_adj[u] & (1 << v)):
                    continue
                L_mask, R_mask = expand_biclique(u, v, rem_adj)
                edges_covered = popcount(L_mask) * popcount(R_mask)
                # surrogate: prefer larger bicliques that also reduce residual
                score = edges_covered
                if score > best_score:
                    best_score = score
                    best_edges = edges_covered
                    best_L = L_mask
                    best_R = R_mask

            if best_L is None:
                for u, v in candidates:
                    if rem_adj[u] & (1 << v):
                        best_L, best_R = expand_biclique(u, v, rem_adj)
                        best_edges = popcount(best_L) * popcount(best_R)
                        break
                if best_L is None:
                    break

            ml = best_L
            while ml:
                lsb = ml & -ml
                u_cur = lsb.bit_length() - 1
                ml ^= lsb
                rem_adj[u_cur] &= ~best_R
            mr = best_R
            while mr:
                lsb = mr & -mr
                v_cur = lsb.bit_length() - 1
                mr ^= lsb
                rem_adj[v_cur] &= ~best_L

            edges_left -= best_edges
            bicliques.append([[x for x in mask_to_list(best_L)],
                              [x for x in mask_to_list(best_R)]])
        return bicliques

    # ---- baseline ----
    rem_mask = adj[:]
    seed_bicliques = []
    edges_left = m
    while edges_left > 0:
        u = None
        for v in range(n):
            if rem_mask[v]:
                u = v
                break
        if u is None:
            break
        mm = rem_mask[u]
        lsb = mm & -mm
        v = lsb.bit_length() - 1
        L, R, _ = expand_biclique(u, v, rem_mask)
        seed_bicliques.append([mask_to_list(L), mask_to_list(R)])
        mask_R = R
        mask_L = L
        for lu in mask_to_list(L):
            rem_mask[lu] &= ~mask_R
        for rv in mask_to_list(R):
            rem_mask[rv] &= ~mask_L
        edges_left -= popcount(L) * popcount(R)

    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # ---- multi-restart greedy ----
    restarts = 200
    for r in range(restarts):
        if time.time() > deadline:
            break
        res = greedy_partition(r)
        if res is None:
            continue
        if not verify(res):
            continue
        if len(res) < best_count:
            best_count = len(res)
            best_bicliques = res
            if best_count == 1:
                break

    # ---- local improvement: matching-inspired edge-disjoint independent set ----
    # Build a pool of maximal biclique candidates from the current partition's
    # residual structure, then select a minimal edge-disjoint cover.
    def refine(bicliques):
        if len(bicliques) <= 1:
            return bicliques
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # (a) try to merge any pair into one valid biclique
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return bicliques
                for j in range(i + 1, len(bicliques)):
                    L_i = set(bicliques[i][0]); R_i = set(bicliques[i][1])
                    L_j = set(bicliques[j][0]); R_j = set(bicliques[j][1])
                    # try L = L_i|L_j, R = R_i|R_j
                    L_new = L_i | L_j
                    R_new = R_i | R_j
                    valid = True
                    for u in L_new:
                        for v in R_new:
                            if not (adj[u] & (1 << v)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and L_new and R_new:
                        new_bicliques = [bicliques[k] for k in range(len(bicliques))
                                         if k != i and k != j]
                        new_bicliques.append([list(L_new), list(R_new)])
                        if verify(new_bicliques):
                            bicliques = new_bicliques
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # (b) re-derive each biclique against residual to shrink redundancy
            # For each biclique i, compute edges it covers, remove them, and
            # check whether the residual can be repartitioned with fewer total.
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return bicliques
                L_i = set(bicliques[i][0]); R_i = set(bicliques[i][1])
                e_i = cross_edges(list_to_mask(L_i), list_to_mask(R_i))
                # build residual edge set
                rem_edges = edge_set - e_i
                # greedy partition of residual
                rem_adj2 = adj[:]
                for (a, b) in rem_edges:
                    pass
                # reconstruct rem_adj by removing e_i edges
                for (a, b) in e_i:
                    rem_adj2[a] &= ~(1 << b)
                    rem_adj2[b] &= ~(1 << a)
                # greedy on residual
                bic2 = []
                edges_left2 = len(rem_edges)
                while edges_left2 > 0:
                    if time.time() > deadline:
                        return bicliques
                    u = None
                    for v in range(n):
                        if rem_adj2[v]:
                            u = v
                            break
                    if u is None:
                        break
                    mm = rem_adj2[u]
                    lsb = mm & -mm
                    v = lsb.bit_length() - 1
                    L, R = expand_biclique(u, v, rem_adj2)
                    bic2.append([mask_to_list(L), mask_to_list(R)])
                    for lu in mask_to_list(L):
                        rem_adj2[lu] &= ~R
                    for rv in mask_to_list(R):
                        rem_adj2[rv] &= ~L
                    edges_left2 -= popcount(L) * popcount(R)
                # candidate = bicliques[i] + bic2
                cand = [bicliques[i]] + bic2
                if verify(cand) and len(cand) < len(bicliques):
                    bicliques = cand
                    improved = True
                    break
        return bicliques

    best_bicliques = refine(best_bicliques)
    best_count = len(best_bicliques)

    # ---- final: try matching-inspired independent set over candidate pool ----
    # Generate pool of maximal biclique candidates; select minimal edge-disjoint
    # cover via greedy (conflict graph = overlapping coverage).
    def independent_set_cover():
        if time.time() > deadline:
            return None
        # pool: each maximal biclique from several random greedy runs' first picks
        pool = []
        for s in range(40):
            if time.time() > deadline:
                break
            rem_adj = adj[:]
            for _ in range(3):
                if time.time() > deadline:
                    break
                active = [u for u in range(n) if rem_adj[u] != 0]
                if not active:
                    break
                u = random.choice(active)
                mm = rem_adj[u]
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                L, R = expand_biclique(u, v, rem_adj)
                e = cross_edges(L, R)
                pool.append((L, R, e))
                for lu in mask_to_list(L):
                    rem_adj[lu] &= ~R
                for rv in mask_to_list(R):
                    rem_adj[rv] &= ~L
        if not pool:
            return None
        # greedy set cover with matching-style tie-break: pick candidate covering
        # most uncovered edges that is edge-disjoint from already selected
        uncovered = set(edge_set)
        selected = []
        while uncovered:
            if time.time() > deadline:
                return None
            best = None
            best_gain = 0
            for (L, R, e) in pool:
                gain = len(e & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best = (L, R, e)
            if best is None or best_gain == 0:
                return None
            L, R, e = best
            selected.append([mask_to_list(L), mask_to_list(R)])
            uncovered -= e
        if verify(selected) and len(selected) < best_count:
            return selected
        return None

    pool_result = independent_set_cover()
    if pool_result is not None:
        best_bicliques = pool_result
        best_count = len(best_bicliques)

    # ---- return with exact types ----
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    return result
