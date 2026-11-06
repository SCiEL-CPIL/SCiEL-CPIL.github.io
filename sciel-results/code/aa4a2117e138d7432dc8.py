import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    n = 18
    all_vertices = list(range(n))

    # ---------- Fast bitmask helpers ----------
    def neighbors_from_bm(bm):
        """Return list of neighbor bitmasks for each vertex."""
        nb = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and (bm >> (i * n + j)) & 1:
                    mask |= (1 << j)
            nb[i] = mask
        return nb

    def tri_conflicts_from_bm(bm):
        """
        For each edge (i,j), return bitmask of vertices k that form a triangle with i,j.
        tri_conf[i][j] = common neighbors of i and j (excluding i, j themselves)
        """
        nb = neighbors_from_bm(bm)
        tc = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (bm >> (i * n + j)) & 1:
                    common = nb[i] & nb[j] & ~(1 << i) & ~(1 << j)
                    tc[i][j] = common
                    tc[j][i] = common
        return tc

    def max_triangle_free_induced_bm(bm, timeout=5.0):
        """
        Exact branch and bound for max triangle-free induced subgraph size.
        A set S is triangle-free if it contains no triangle.
        We build S incrementally; when we add v to current set C,
        we must ensure no triangle is formed. A triangle is formed if there
        exist u, w in C such that v-u, v-w, u-w are all edges.
        Equivalently, for each pair (u,w) in C that is an edge, v must not
        be in tri_conf[u][w].
        """
        best = [0]
        start = time.time()
        tc = tri_conflicts_from_bm(bm)

        def dfs(current_mask, candidates_mask):
            if time.time() - start > timeout:
                return
            cur_size = bin(current_mask).count('1')
            cand_size = bin(candidates_mask).count('1')
            if cur_size + cand_size <= best[0]:
                return
            if cand_size == 0:
                if cur_size > best[0]:
                    best[0] = cur_size
                return

            # Pick first candidate
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            remaining = candidates_mask & ~v_bit

            # Check if v can be added
            can_add = True
            cur_list = [i for i in range(n) if (current_mask >> i) & 1]
            for idx in range(len(cur_list)):
                for jdx in range(idx + 1, len(cur_list)):
                    if tc[cur_list[idx]][cur_list[jdx]] & v_bit:
                        can_add = False
                        break
                if not can_add:
                    break

            if can_add:
                new_current = current_mask | v_bit
                # Filter candidates: w can remain if for all pairs in new_current,
                # w is not in their tri_conflict
                new_remaining = 0
                new_cur_list = cur_list + [v]
                for w in range(n):
                    if (remaining >> w) & 1:
                        w_bit = 1 << w
                        ok = True
                        for a in range(len(new_cur_list)):
                            for b in range(a + 1, len(new_cur_list)):
                                if tc[new_cur_list[a]][new_cur_list[b]] & w_bit:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            new_remaining |= w_bit
                dfs(new_current, new_remaining)

            # Skip v
            dfs(current_mask, remaining)

        dfs(0, (1 << n) - 1)
        return best[0]

    def greedy_triangle_free_size_bm(bm, trials=20):
        """Quick greedy estimate."""
        nb = neighbors_from_bm(bm)
        best = 0
        for _ in range(trials):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                ok = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        if (nb[v] >> current[i]) & 1 and (nb[v] >> current[j]) & 1 and \
                           (nb[current[i]] >> current[j]) & 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    current.append(v)
            best = max(best, len(current))
        return best

    # ---------- K4 check ----------
    def is_k4_free_bm(bm):
        """Check K4-freeness. For each 4-set, check if all 6 edges present."""
        for a, b, c, d in combinations(range(n), 4):
            if ((bm >> (a*n+b)) & 1) and ((bm >> (a*n+c)) & 1) and ((bm >> (a*n+d)) & 1) and \
               ((bm >> (b*n+c)) & 1) and ((bm >> (b*n+d)) & 1) and ((bm >> (c*n+d)) & 1):
                return False
        return True

    def try_add_edge(bm, u, v):
        """Return new bm if adding (u,v) keeps K4-free, else None."""
        if (bm >> (u * n + v)) & 1:
            return None
        new_bm = bm | (1 << (u * n + v)) | (1 << (v * n + u))
        # Check: common neighbors of u and v that form an edge -> K4
        nb_u = 0
        nb_v = 0
        for w in range(n):
            if w != u and w != v:
                if (bm >> (u * n + w)) & 1:
                    nb_u |= (1 << w)
                if (bm >> (v * n + w)) & 1:
                    nb_v |= (1 << w)
        common = nb_u & nb_v
        cl = [w for w in range(n) if (common >> w) & 1]
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if (bm >> (cl[i] * n + cl[j])) & 1:
                    return None
        return new_bm

    def try_remove_edge(bm, u, v):
        """Return new bm with edge (u,v) removed."""
        if not ((bm >> (u * n + v)) & 1):
            return None
        return bm & ~((1 << (u * n + v)) | (1 << (v * n + u)))

    # ---------- Initial construction: K6,6,6 ----------
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    bm = 0
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    bm |= (1 << (a * n + b)) | (1 << (b * n + a))

    current_bm = bm
    current_score = max_triangle_free_induced_bm(current_bm, timeout=3.0)
    best_bm = current_bm
    best_score = current_score

    # ---------- Phase 1: Simulated Annealing with bitmask ----------
    T = 3.0
    cooling = 0.998
    no_improve = 0
    edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    while time.time() < deadline - 15:
        # Pick random edge
        u, v = random.choice(edge_pairs)
        if (current_bm >> (u * n + v)) & 1:
            new_bm = try_remove_edge(current_bm, u, v)
        else:
            new_bm = try_add_edge(current_bm, u, v)

        if new_bm is None:
            T *= cooling
            no_improve += 1
            continue

        # Quick greedy check
        g_score = greedy_triangle_free_size_bm(new_bm, trials=5)
        delta_g = g_score - current_score

        if delta_g < 0 or random.random() < np.exp(-delta_g / max(T, 0.01)):
            exact_score = max_triangle_free_induced_bm(new_bm, timeout=1.5)
            delta = exact_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                current_bm = new_bm
                current_score = exact_score
                no_improve = 0
                if exact_score < best_score:
                    best_bm = new_bm
                    best_score = exact_score
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling
        if no_improve > 150:
            T = max(T, 0.8)
            no_improve = 0

    # ---------- Phase 2: Focused local search on best graph ----------
    # Try all single-edge flips that improve, iteratively
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for u, v in edge_pairs:
            if time.time() > deadline - 10:
                break
            if (best_bm >> (u * n + v)) & 1:
                cand_bm = try_remove_edge(best_bm, u, v)
            else:
                cand_bm = try_add_edge(best_bm, u, v)
            if cand_bm is None:
                continue
            cand_score = max_triangle_free_induced_bm(cand_bm, timeout=1.0)
            if cand_score < best_score:
                best_bm = cand_bm
                best_score = cand_score
                improved = True

    # ---------- Phase 3: Multi-start restarts if time allows ----------
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            # New random start: random K4-free graph
            rbm = 0
            for u, v in edge_pairs:
                if random.random() < 0.4:
                    new_bm = try_add_edge(rbm, u, v)
                    if new_bm is not None:
                        rbm = new_bm
            r_score = max_triangle_free_induced_bm(rbm, timeout=2.0)
            if r_score < best_score:
                best_bm = rbm
                best_score = r_score
            # Short SA from this start
            T2 = 2.0
            for _ in range(500):
                if time.time() > deadline - 15:
                    break
                u, v = random.choice(edge_pairs)
                if (rbm >> (u * n + v)) & 1:
                    nbm = try_remove_edge(rbm, u, v)
                else:
                    nbm = try_add_edge(rbm, u, v)
                if nbm is None:
                    continue
                gs = greedy_triangle_free_size_bm(nbm, trials=3)
                dg = gs - r_score
                if dg < 0 or random.random() < np.exp(-dg / max(T2, 0.01)):
                    es = max_triangle_free_induced_bm(nbm, timeout=1.0)
                    d = es - r_score
                    if d < 0 or random.random() < np.exp(-d / max(T2, 0.01)):
                        rbm = nbm
                        r_score = es
                        if es < best_score:
                            best_bm = nbm
                            best_score = es
                T2 *= 0.995

    # ---------- Final exact verification ----------
    final_score = max_triangle_free_induced_bm(best_bm, timeout=8.0)
    if final_score < best_score:
        best_score = final_score

    # ---------- Convert to edge list ----------
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_bm >> (i * n + j)) & 1:
                edges.append([i, j])
    return edges
