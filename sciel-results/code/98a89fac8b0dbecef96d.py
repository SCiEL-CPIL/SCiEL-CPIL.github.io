import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s
    sys.setrecursionlimit(100000)

    # ---------- Fallback: 10 disjoint triangles (n=30, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- C4-free check (bitmask) ----------
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):  # more than one common neighbor
                    return False
        return True

    # ---------- Targeted independent-set search (bitmask, branch & bound) ----------
    def has_independent_set(adj_mask, n, target=11, node_cap=2000000):
        node_count = [0]

        def dfs(candidates, size):
            if size >= target:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target:
                return False

            node_count[0] += 1
            if node_count[0] > node_cap:
                return False

            # Pick vertex with highest degree in candidate subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    if deg >= 10:
                        break

            if best_v == -1:
                return False

            # Include best_v
            new_cands = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_cands, size + 1):
                return True

            # Exclude best_v
            new_cands = candidates & ~(1 << best_v)
            if dfs(new_cands, size):
                return True

            return False

        all_cands = (1 << n) - 1
        return dfs(all_cands, 0)

    # ---------- Greedy C4-free graph generation (per-vertex incremental) ----------
    def generate_greedy_c4_free(n, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n

        for i in range(n):
            candidates = list(range(i))
            rng.shuffle(candidates)
            for j in candidates:
                # Check if adding (i, j) creates a C4:
                # For all v in N(j): N(i) & N(v) must be empty
                v_mask = adj_mask[j]
                can_add = True
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                if can_add:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    # ---------- Repair / local-search: add edges to reduce alpha ----------
    def repair(adj_mask, n, seed, time_budget):
        rng = random.Random(seed)
        t_end = time.time() + time_budget
        # Build list of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj_mask[u] >> v) & 1):
                    non_edges.append((u, v))
        rng.shuffle(non_edges)

        improved = False
        for (u, v) in non_edges:
            if time.time() > t_end:
                break
            # Check if adding (u, v) creates a C4
            nu = adj_mask[u]
            nv = adj_mask[v]
            # For all x in N(u): N(x) & N(v) must be empty
            c = nu
            safe = True
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nv:
                    safe = False
                    break
            if not safe:
                continue
            # Try adding
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            improved = True

        return improved

    # ---------- Build edge list from bitmask ----------
    def build_edges(n, adj_mask):
        edges = []
        for u in range(n):
            m = adj_mask[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # ---------- Search from high n down ----------
    best = None
    for n in range(43, 29, -1):
        if time.time() > deadline - 5:
            break
        seeds = 0
        max_seeds = 5000
        found = False
        while time.time() < deadline - 5 and seeds < max_seeds:
            adj_mask = generate_greedy_c4_free(n, seeds)

            # Check independence
            if not has_independent_set(adj_mask, n, 11):
                edges = build_edges(n, adj_mask)
                best = {'n': n, 'edges': edges}
                found = True
                break

            # If close (alpha is 11 or 12), try a repair pass
            # We don't know alpha exactly, but if the search didn't find 11
            # quickly it might be 12. Try repair with a small time budget.
            if seeds % 5 == 0:
                adj_mask2 = [m for m in adj_mask]  # copy
                t0 = time.time()
                repair(adj_mask2, n, seeds * 7 + 1, 0.05)
                if time.time() - t0 < 0.05:  # only if repair was fast
                    if not has_independent_set(adj_mask2, n, 11):
                        edges = build_edges(n, adj_mask2)
                        best = {'n': n, 'edges': edges}
                        found = True
                        break

            seeds += 1

        if found:
            break

    if best is None:
        best = fallback()

    return best
