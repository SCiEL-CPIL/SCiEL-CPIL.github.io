"""Rewritten program: C4-minimizing 91-edge graph on 31 vertices."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np

    start = time.time()
    DEADLINE = 148.0  # 150s wall-clock limit, 2s safety
    n = 31
    target_edges = 91

    # ---- Build the polarity graph of PG(2,5): 31 vertices, 93 edges, 45 C4 ----
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    poly_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                poly_edges.add((i, j))

    # ---- Fast C4 counting via common-neighbor pairs ----
    # C4 count = sum over all unordered vertex pairs (u,v) of C(common(u,v), 2)
    # Each C4 has exactly 2 pairs of opposite vertices, each contributing 1.
    def count_c4(adj_sets):
        total = 0
        for u in range(n):
            nu = adj_sets[u]
            for v in range(u + 1, n):
                c = len(nu & adj_sets[v])
                total += c * (c - 1) // 2
        return total

    # ---- Incremental C4 delta for adding/removing a single edge ----
    # When adding edge (u,v): new C4s = number of edges among common neighbors of u,v
    # (excluding u,v themselves, which they aren't since graph is simple)
    # When removing edge (u,v): destroyed C4s = same count
    def c4_delta_edge(u, v, adj_sets):
        """Return the number of C4s that would be created (or destroyed) by edge (u,v)."""
        cn = adj_sets[u] & adj_sets[v]
        cnt = 0
        cn_list = list(cn)
        for i in range(len(cn_list)):
            a = cn_list[i]
            for j in range(i + 1, len(cn_list)):
                b = cn_list[j]
                if b in adj_sets[a]:
                    cnt += 1
        return cnt

    # ---- Initial state: remove 2 edges from polarity graph ----
    # The polarity graph is (31, 6, 2, 1) strongly regular: 93 edges, 45 C4.
    # Removing 2 edges: each edge is in 2 C4s (since lambda=2), but two edges
    # may share C4s.  Try all C(93,2)=4278 pairs and pick the best.
    poly_edge_list = list(poly_edges)
    m = len(poly_edge_list)

    # Build initial adjacency sets
    adj0 = [set() for _ in range(n)]
    for u, v in poly_edges:
        adj0[u].add(v)
        adj0[v].add(u)

    # Precompute c4_delta for each edge in the polarity graph
    edge_delta = {}
    for (u, v) in poly_edge_list:
        edge_delta[(u, v)] = c4_delta_edge(u, v, adj0)

    # For each pair of edges to remove, compute resulting C4 count
    # C4_after = 45 - d1 - d2 + shared_c4s(e1, e2)
    # shared_c4s: C4s containing both e1 and e2
    # For a C4 to contain both edges, the 4 edges of the C4 must include e1 and e2.
    # If e1=(a,b) and e2=(c,d) are disjoint: C4 must be a-c-b-d-a or a-d-b-c-a
    #   Check: a-c edge, c-b edge, b-d edge, d-a edge all present
    # If e1 and e2 share a vertex, say e1=(a,b), e2=(a,c):
    #   C4 must be b-x-a-c-b or b-x-a-c... wait, need 4 distinct vertices.
    #   C4: b - x - a - c - b? No, that's b-x, x-a, a-c, c-b.
    #   Or: b - c - x - a - b? b-c, c-x, x-a, a-b.
    #   Actually with shared vertex a: C4 uses vertices {a, b, c, x}
    #   Edges needed: a-b (e1), a-c (e2), and two more from {b,c,x}
    #   C4: b-x-c-a-b needs b-x, x-c, c-a(=e2), a-b(=e1) → check b-x and x-c
    #   C4: b-c-x-a-b needs b-c, c-x, x-a, a-b → check b-c, c-x, x-a
    #   C4: b-x-a-c-b needs b-x, x-a, a-c(=e2), c-b → check b-x, x-a, c-b
    #   These overlap. Let me just check all 4-cycles containing both edges directly.

    def shared_c4s(e1, e2, adj_sets):
        """Count C4s containing both edges e1 and e2."""
        a, b = e1
        c, d = e2
        # Collect the set of vertices involved
        verts = {a, b, c, d}
        if len(verts) < 4:
            # Shared vertex: find the 4th vertex
            # C4 must use 4 distinct vertices
            # Try all possible 4th vertices
            count = 0
            for x in range(n):
                if x in verts:
                    continue
                # Check if {a,b,c,x} forms a C4 containing both e1 and e2
                # e1=(a,b), e2=(c,d) where one of c,d == a or b
                # The C4 on vertices {a,b,c,x} (where d is one of a,b) must contain edges a-b and a-c (or b-c)
                # A C4 on 4 vertices has 4 edges. We need a-b and a-c (or b-c) among them.
                # The C4 must be a 4-cycle: two opposite pairs.
                # For vertices {a,b,c,x} with edges a-b and a-c:
                #   a is adjacent to both b and c in the C4, so a is not opposite to either.
                #   a is opposite to x. b and c are opposite.
                #   C4: a-b-x-c-a: needs a-b, b-x, x-c, c-a ✓
                #   C4: a-c-x-b-a: needs a-c, c-x, x-b, b-a ✓ (same cycle, different direction)
                #   So check: b-x edge AND x-c edge
                if b in adj_sets[x] and c in adj_sets[x]:
                    count += 1
                # Also: a-b-x-c-a is the only C4 with a opposite x and b opposite c
                # What about a-c-x-b-a? Same cycle.
                # What if e2 = (b,c) instead of (a,c)?
                # Then C4: a-b-c-x-a: needs a-b, b-c, c-x, x-a
                #   Check c-x and x-a
                # This is a different C4 structure.
            return count
        else:
            # Disjoint edges: C4 must be a-c-b-d-a or a-d-b-c-a
            # Check a-c, c-b, b-d, d-a
            if (c in adj_sets[a] and b in adj_sets[c] and d in adj_sets[b] and a in adj_sets[d]):
                return 1
            if (d in adj_sets[a] and b in adj_sets[d] and c in adj_sets[b] and a in adj_sets[c]):
                return 1
            return 0

    # This shared_c4s function is getting complex. Let me use a simpler approach:
    # Just compute C4 count directly for the best few candidates.
    # Actually, for the initial step, let me just try removing pairs and count directly.
    # 4278 pairs × O(n^2) counting = 4278 × ~500 ops ≈ 2M ops. Fast enough.

    best_init_c4 = float('inf')
    best_init_removed = None

    # Precompute for speed: for each edge, its delta
    # For pair (e1, e2): C4 = 45 - d1 - d2 + shared
    # But shared is hard. Let me just directly count for top candidates.
    # First, compute d1 + d2 for all pairs, sort, try top 100 with direct count.

    pair_scores = []
    for i in range(m):
        e1 = poly_edge_list[i]
        d1 = edge_delta[e1]
        for j in range(i + 1, m):
            e2 = poly_edge_list[j]
            d2 = edge_delta[e2]
            pair_scores.append((d1 + d2, i, j))

    pair_scores.sort()

    # Try top 200 pairs with direct C4 count
    for score, i, j in pair_scores[:200]:
        if time.time() - start > DEADLINE - 10:
            break
        e1 = poly_edge_list[i]
        e2 = poly_edge_list[j]
        removed = {e1, e2}
        adj_tmp = [set() for _ in range(n)]
        for u, v in poly_edges:
            if (u, v) not in removed and (v, u) not in removed:
                adj_tmp[u].add(v)
                adj_tmp[v].add(u)
        c4 = count_c4(adj_tmp)
        if c4 < best_init_c4:
            best_init_c4 = c4
            best_init_removed = (e1, e2)

    # Build initial state
    if best_init_removed is None:
        best_init_removed = (poly_edge_list[0], poly_edge_list[1])
        adj = [set() for _ in range(n)]
        for u, v in poly_edges:
            if (u, v) != best_init_removed[0] and (u, v) != best_init_removed[1]:
                adj[u].add(v)
                adj[v].add(u)
        current_c4 = count_c4(adj)
    else:
        e1, e2 = best_init_removed
        adj = [set() for _ in range(n)]
        for u, v in poly_edges:
            if (u, v) != e1 and (u, v) != e2:
                adj[u].add(v)
                adj[v].add(u)
        current_c4 = best_init_c4

    current_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                current_edges.add((u, v))

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = set(all_pairs) - current_edges

    # ---- Simulated Annealing with incremental deltas ----
    # Key insight: for a 1-edge swap (remove e1, add e2):
    #   new_c4 = current_c4 - delta_remove(e1) + delta_add(e2_after_removal)
    #   But delta_add depends on the graph AFTER removing e1, which is complex.
    #   However, for the polarity graph (and its small perturbations), the deltas
    #   are small (0-4), so we can approximate or compute exactly.
    #
    # Exact computation:
    #   delta_remove(e1) = c4_delta_edge(e1, adj)  [C4s destroyed by removing e1]
    #   After removing e1, compute delta_add(e2) in the new graph.
    #   This requires modifying adj temporarily.

    def swap_delta(e_remove, e_add, adj_sets, current_c4_val):
        """Compute new C4 after removing e_remove and adding e_add."""
        u1, v1 = e_remove
        u2, v2 = e_add

        # C4s destroyed by removing (u1,v1)
        d_rem = c4_delta_edge(u1, v1, adj_sets)

        # Temporarily remove (u1,v1)
        adj_sets[u1].discard(v1)
        adj_sets[v1].discard(u1)

        # C4s created by adding (u2,v2) in the modified graph
        d_add = c4_delta_edge(u2, v2, adj_sets)

        # Restore
        adj_sets[u1].add(v1)
        adj_sets[v1].add(u1)

        return current_c4_val - d_rem + d_add

    rng = random.Random(42)
    best_c4 = current_c4
    best_edges = set(current_edges)

    # Annealing schedule
    T = 3.0
    T_min = 0.1
    cooling = 0.9995
    max_steps = 500000

    step = 0
    no_improve = 0
    improve_threshold = 5000

    while step < max_steps and time.time() - start < DEADLINE - 3:
        # Pick a random edge to remove
        e_remove = rng.choice(list(current_edges))
        # Pick a random non-edge to add
        e_add = rng.choice(list(non_edges))

        new_c4 = swap_delta(e_remove, e_add, adj, current_c4)

        if new_c4 < current_c4:
            # Accept (improvement)
            u1, v1 = e_remove
            u2, v2 = e_add
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            current_edges.discard(e_remove)
            current_edges.add(e_add)
            non_edges.discard(e_add)
            non_edges.add(e_remove)
            current_c4 = new_c4
            no_improve = 0
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(current_edges)
        elif T > T_min:
            # Metropolis acceptance
            delta = new_c4 - current_c4
            if delta <= 0:
                accept = True
            else:
                # Avoid overflow: delta is small (0-10), T >= 0.1
                if delta / T > 50:
                    accept = False
                else:
                    accept = rng.random() < np.exp(-delta / T)
            if accept:
                u1, v1 = e_remove
                u2, v2 = e_add
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                current_edges.discard(e_remove)
                current_edges.add(e_add)
                non_edges.discard(e_add)
                non_edges.add(e_remove)
                current_c4 = new_c4
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling
        step += 1

        # Periodic restart if stuck
        if no_improve > improve_threshold and T > T_min:
            # Reset to best
            if current_c4 > best_c4:
                adj = [set() for _ in range(n)]
                current_edges = set(best_edges)
                for u, v in current_edges:
                    adj[u].add(v)
                    adj[v].add(u)
                non_edges = set(all_pairs) - current_edges
                current_c4 = best_c4
            T = max(T, 1.0)
            no_improve = 0

    # ---- Second phase: greedy hill-climbing from best state ----
    if time.time() - start < DEADLINE - 5:
        adj = [set() for _ in range(n)]
        current_edges = set(best_edges)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        non_edges = set(all_pairs) - current_edges
        current_c4 = best_c4

        improved = True
        while improved and time.time() - start < DEADLINE - 3:
            improved = False
            best_swap = None
            best_new_c4 = current_c4

            # Try all edges for removal, sample non-edges for addition
            edges_list = list(current_edges)
            non_edges_list = list(non_edges)
            rng2 = random.Random(99)

            for e_remove in edges_list:
                if time.time() - start > DEADLINE - 3:
                    break
                u1, v1 = e_remove
                d_rem = c4_delta_edge(u1, v1, adj)

                # Try a subset of non-edges
                candidates = rng2.sample(non_edges_list, min(30, len(non_edges_list)))
                for e_add in candidates:
                    if e_add == e_remove:
                        continue
                    u2, v2 = e_add
                    # Compute delta_add after removing e_remove
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    d_add = c4_delta_edge(u2, v2, adj)
                    adj[u1].add(v1)
                    adj[v1].add(u1)

                    new_c4 = current_c4 - d_rem + d_add
                    if new_c4 < best_new_c4:
                        best_new_c4 = new_c4
                        best_swap = (e_remove, e_add)

            if best_swap:
                e_remove, e_add = best_swap
                u1, v1 = e_remove
                u2, v2 = e_add
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                current_edges.discard(e_remove)
                current_edges.add(e_add)
                non_edges.discard(e_add)
                non_edges.add(e_remove)
                current_c4 = best_new_c4
                improved = True
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = set(current_edges)

    # ---- Final: try 2-edge swaps if time permits ----
    if time.time() - start < DEADLINE - 8:
        adj = [set() for _ in range(n)]
        current_edges = set(best_edges)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        non_edges = set(all_pairs) - current_edges
        current_c4 = best_c4

        rng3 = random.Random(7)
        for _ in range(5000):
            if time.time() - start > DEADLINE - 3:
                break
            e1, e2 = rng3.sample(list(current_edges), 2)
            e3, e4 = rng3.sample(list(non_edges), 2)

            # Apply 2-edge swap
            for (a, b) in (e1, e2):
                adj[a].discard(b)
                adj[b].discard(a)
            for (a, b) in (e3, e4):
                adj[a].add(b)
                adj[b].add(a)

            new_c4 = count_c4(adj)

            # Revert
            for (a, b) in (e1, e2):
                adj[a].add(b)
                adj[b].add(a)
            for (a, b) in (e3, e4):
                adj[a].discard(b)
                adj[b].discard(a)

            if new_c4 < current_c4:
                for (a, b) in (e1, e2):
                    adj[a].discard(b)
                    adj[b].discard(a)
                for (a, b) in (e3, e4):
                    adj[a].add(b)
                    adj[b].add(a)
                current_edges.discard(e1)
                current_edges.discard(e2)
                current_edges.add(e3)
                current_edges.add(e4)
                non_edges.discard(e3)
                non_edges.discard(e4)
                non_edges.add(e1)
                non_edges.add(e2)
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = set(current_edges)

    # ---- Return best result ----
    result_edges = best_edges
    # Safety: ensure exactly 91 edges
    if len(result_edges) != target_edges:
        # Fallback: rebuild from polarity graph minus 2 edges
        result_edges = set(poly_edges) - {best_init_removed[0], best_init_removed[1]}
        if len(result_edges) > target_edges:
            result_edges = set(list(result_edges)[:target_edges])
        elif len(result_edges) < target_edges:
            for p in all_pairs:
                if len(result_edges) >= target_edges:
                    break
                if p not in result_edges:
                    result_edges.add(p)

    return [[u, v] for u, v in sorted(result_edges)]
# EVOLVE-BLOCK-END
