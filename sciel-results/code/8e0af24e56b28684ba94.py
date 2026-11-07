import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))

    all_edges = list(edge_set)
    m = len(all_edges)

    # Helper: get list of vertices from mask
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    # Helper: compute cross edges of a biclique as a set
    def cross_edges(L_mask, R_mask):
        es = set()
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            rm = R_mask
            while rm:
                low_r = rm & (-rm)
                v = low_r.bit_length() - 1
                es.add((min(u, v), max(u, v)))
                rm ^= low_r
            lm ^= low
        return es

    # Check if (L_mask, R_mask) is a valid biclique: all cross pairs are graph edges
    def is_valid_biclique(L_mask, R_mask):
        if L_mask == 0 or R_mask == 0:
            return False
        lm = L_mask
        while lm:
            low = lm & (-lm)
            u = low.bit_length() - 1
            if (adj_mask[u] & R_mask) != R_mask:
                return False
            lm ^= low
        return True

    # Expand a biclique to be maximal w.r.t. remaining edges
    # A vertex can be added to L if it's connected to all of R and all those edges are in remaining
    def expand_biclique(u, v, remaining):
        L_mask = 1 << u
        R_mask = 1 << v

        improved = True
        while improved:
            improved = False

            # Try adding to L
            for x in range(n):
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue

                # Check if x can join L: x must be connected to all r in R, and all (x,r) in remaining
                ok = True
                rm = R_mask
                while rm:
                    low_r = rm & (-rm)
                    r = low_r.bit_length() - 1
                    if not ((adj_mask[x] >> r) & 1):
                        ok = False
                        break
                    e = (min(x, r), max(x, r))
                    if e not in remaining:
                        ok = False
                        break
                    rm ^= low_r

                if ok:
                    L_mask |= (1 << x)
                    improved = True

            # Try adding to R
            for y in range(n):
                if (L_mask >> y) & 1 or (R_mask >> y) & 1:
                    continue

                # Check if y can join R: y must be connected to all l in L, and all (l,y) in remaining
                ok = True
                lm = L_mask
                while lm:
                    low_l = lm & (-lm)
                    l = low_l.bit_length() - 1
                    if not ((adj_mask[y] >> l) & 1):
                        ok = False
                        break
                    e = (min(l, y), max(l, y))
                    if e not in remaining:
                        ok = False
                        break
                    lm ^= low_l

                if ok:
                    R_mask |= (1 << y)
                    improved = True

        return L_mask, R_mask

    # Greedy set-cover: repeatedly pick the biclique that covers the most uncovered edges
    def greedy_set_cover():
        remaining = set(all_edges)
        bicliques = []

        while remaining:
            if time.time() > deadline - 2:
                break

            best_L = 0
            best_R = 0
            best_count = 0

            # Try expanding from several seed edges
            # Use deterministic ordering for reproducibility, but try multiple seeds
            seed_edges = list(remaining)
            # Sort by degree sum for better heuristic
            seed_edges.sort(key=lambda e: -((adj_mask[e[0]] & adj_mask[e[1]]).bit_count()))

            # Try up to 40 seed edges
            for e in seed_edges[:40]:
                L_mask, R_mask = expand_biclique(e[0], e[1], remaining)
                ce = cross_edges(L_mask, R_mask)
                # Count how many of these are in remaining
                cnt = sum(1 for x in ce if x in remaining)
                if cnt > best_count:
                    best_count = cnt
                    best_L = L_mask
                    best_R = R_mask

            if best_L == 0:
                # Fallback
                e = next(iter(remaining))
                best_L = 1 << e[0]
                best_R = 1 << e[1]
                best_count = 1

            L_list = mask_to_list(best_L)
            R_list = mask_to_list(best_R)
            bicliques.append([L_list, R_list])

            # Remove covered edges
            ce = cross_edges(best_L, best_R)
            for e in ce:
                remaining.discard(e)

        return bicliques

    # Local search: try to merge pairs of bicliques into a single biclique
    def local_search(bicliques):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            nb = len(bicliques)
            for i in range(nb):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, nb):
                    if time.time() > deadline - 3:
                        break

                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]

                    # Try merging: (Li ∪ Lj, Ri ∩ Rj) and (Li ∩ Lj, Ri ∪ Rj)
                    # and also try finding a biclique that covers the union of edges
                    # from both bicliques

                    # Compute edges covered by each
                    ce_i = set()
                    for u in Li:
                        for v in Ri:
                            ce_i.add((min(u, v), max(u, v)))

                    ce_j = set()
                    for u in Lj:
                        for v in Rj:
                            ce_j.add((min(u, v), max(u, v)))

                    # Try to find a biclique that covers ce_i ∪ ce_j
                    union_edges = ce_i | ce_j
                    vertices = set()
                    for u, v in union_edges:
                        vertices.add(u)
                        vertices.add(v)

                    # Only attempt if vertex set is small enough
                    if len(vertices) <= 12:
                        vertex_list = sorted(vertices)
                        # Try each vertex as L-side starter
                        for start in vertex_list:
                            L_mask = 1 << start
                            # Find all neighbors of start that are in vertices
                            candidates = [v for v in vertex_list if v != start and (adj_mask[start] >> v) & 1]

                            # Try to build a biclique with start on L
                            # Use greedy expansion
                            R_mask = 0
                            for c in candidates:
                                # Check if c can be added to R
                                ok = True
                                lm = L_mask
                                while lm:
                                    low_l = lm & (-lm)
                                    l = low_l.bit_length() - 1
                                    if not ((adj_mask[c] >> l) & 1):
                                        ok = False
                                        break
                                    lm ^= low_l
                                if ok:
                                    R_mask |= (1 << c)

                            if R_mask == 0:
                                continue

                            # Expand L
                            improved_exp = True
                            while improved_exp:
                                improved_exp = False
                                for x in vertex_list:
                                    if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                                        continue
                                    ok = True
                                    rm = R_mask
                                    while rm:
                                        low_r = rm & (-rm)
                                        r = low_r.bit_length() - 1
                                        if not ((adj_mask[x] >> r) & 1):
                                            ok = False
                                            break
                                        rm ^= low_r
                                    if ok:
                                        L_mask |= (1 << x)
                                        improved_exp = True

                            # Check if this biclique covers all union_edges
                            ce_new = cross_edges(L_mask, R_mask)
                            if union_edges.issubset(ce_new) and ce_new.issubset(ce_i | ce_j):
                                # Valid merge
                                new_bicliques = []
                                for k in range(nb):
                                    if k != i and k != j:
                                        new_bicliques.append(bicliques[k])
                                L_list = mask_to_list(L_mask)
                                R_list = mask_to_list(R_mask)
                                new_bicliques.append([L_list, R_list])
                                bicliques = new_bicliques
                                improved = True
                                break

                            # Also try with start on R
                            R_mask2 = 1 << start
                            L_mask2 = 0
                            for c in candidates:
                                ok = True
                                rm = R_mask2
                                while rm:
                                    low_r = rm & (-rm)
                                    r = low_r.bit_length() - 1
                                    if not ((adj_mask[c] >> r) & 1):
                                        ok = False
                                        break
                                    rm ^= low_r
                                if ok:
                                    L_mask2 |= (1 << c)

                            if L_mask2 == 0:
                                continue

                            improved_exp = True
                            while improved_exp:
                                improved_exp = False
                                for y in vertex_list:
                                    if (L_mask2 >> y) & 1 or (R_mask2 >> y) & 1:
                                        continue
                                    ok = True
                                    lm = L_mask2
                                    while lm:
                                        low_l = lm & (-lm)
                                        l = low_l.bit_length() - 1
                                        if not ((adj_mask[y] >> l) & 1):
                                            ok = False
                                            break
                                        lm ^= low_l
                                    if ok:
                                        R_mask2 |= (1 << y)
                                        improved_exp = True

                            ce_new2 = cross_edges(L_mask2, R_mask2)
                            if union_edges.issubset(ce_new2) and ce_new2.issubset(ce_i | ce_j):
                                new_bicliques = []
                                for k in range(nb):
                                    if k != i and k != j:
                                        new_bicliques.append(bicliques[k])
                                L_list = mask_to_list(L_mask2)
                                R_list = mask_to_list(R_mask2)
                                new_bicliques.append([L_list, R_list])
                                bicliques = new_bicliques
                                improved = True
                                break

                        if improved:
                            break

        return bicliques

    best_bicliques = None
    best_count = float('inf')

    # Multiple restarts with different random seeds
    for restart in range(80):
        if time.time() > deadline - 5:
            break

        random.seed(restart * 12345 + 67890)
        bicliques = greedy_set_cover()

        if not bicliques:
            continue

        bicliques = local_search(bicliques)

        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques

            if best_count == 1:
                break

    if best_bicliques is None:
        # Fallback: single edges
        best_bicliques = [[[u], [v]] for u, v in all_edges]

    # Final conversion to ensure types are correct
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])

    return result

# EVOLVE-BLOCK-END
