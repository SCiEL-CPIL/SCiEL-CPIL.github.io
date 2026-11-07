"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from collections import defaultdict, deque

    start_time = time.time()
    DEADLINE = 135.0  # Leave 15s safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # Cross adjacency: for each independent vertex v, which clique vertices it connects to
        cross_adj = {}
        for v in I:
            cross_adj[v] = adj[v] & set(C)

        # We want to form triangles (v, u, w) where v in I, u,w in C, u<w
        # and (u,w) is an edge (always true since C is clique) and v connected to both u,w
        #
        # Strategy: For each v in I, we have a set S_v of clique vertices it connects to.
        # We want to partition the cross-edges into triangles and pairs.
        # A triangle uses 2 cross-edges (v-u, v-w) and 1 inner edge (u-w).
        # A pair uses 1 cross-edge (v-u) and no inner edge.
        # Inner edges not used in triangles become pairs.
        #
        # Total cliques = (#triangles) + (#cross pairs) + (#unused inner edge pairs)
        # #triangles + #cross_pairs = sum over v of (number of cliques covering v's cross edges)
        # For vertex v with d_v cross edges: if we pack t_v triangles using v, remaining d_v - 2*t_v are pairs
        # So cliques involving v = t_v + (d_v - 2*t_v) = d_v - t_v
        # Total = sum(d_v) - sum(t_v) + (#unused inner edges)
        # sum(d_v) = total cross edges
        # #unused inner edges = C(k,2) - (#inner edges used in triangles) = C(k,2) - sum(t_v) since each triangle uses exactly 1 inner edge
        # Total = sum(d_v) - sum(t_v) + C(k,2) - sum(t_v) = sum(d_v) + C(k,2) - 2*sum(t_v)
        #
        # So minimizing total = maximizing sum(t_v) = total number of triangles!
        #
        # This is equivalent to: find maximum set of vertex-disjoint-in-I, edge-disjoint-in-C triangles.
        # Where triangle = (v, u, w), v in I, {u,w} edge in C-clique (always), v adjacent to u and w.
        #
        # This is a 3D matching / 3D packing problem. Greedy + local search.

        total_cross = sum(len(cross_adj[v]) for v in I)
        total_inner = k * (k - 1) // 2

        def count_triangles(triangles):
            """triangles: list of (v, u, w) where v in I, u,w in C"""
            return len(triangles)

        # Greedy: for each v, greedily pick pairs from cross_adj[v]
        # Use a randomized approach with multiple restarts

        def greedy_solve(order_I, order_C, rng):
            """
            order_I: permutation of I vertices
            order_C: permutation of C vertices (for tie-breaking in matching)
            Returns: list of triangles
            """
            # remaining cross edges per v
            remaining = {v: set(cross_adj[v]) for v in I}
            # remaining inner edges (u, w) with u < w
            used_inner = set()
            triangles = []

            for v in order_I:
                if time.time() > DEADLINE:
                    break
                S = remaining[v]
                if len(S) < 2:
                    continue
                # Find maximum matching in the graph on S (all pairs are edges since C is clique)
                # But we need edge-disjoint in C: no inner edge can be used twice
                # So we need max matching in S avoiding used_inner
                # Since C is a clique, the graph on S is complete minus used_inner edges
                # Max matching in near-complete graph on |S| vertices
                S_list = sorted(S, key=lambda x: order_C.index(x))
                m = len(S_list)
                # Greedy matching: pick edges in order_C preference
                matched = []
                used_in_matching = set()
                for i in range(m):
                    if time.time() > DEADLINE:
                        break
                    u = S_list[i]
                    if u in used_in_matching:
                        continue
                    # Find best w
                    for j in range(i + 1, m):
                        w = S_list[j]
                        if w in used_in_matching:
                            continue
                        key = (min(u, w), max(u, w))
                        if key not in used_inner:
                            matched.append((u, w))
                            used_in_matching.add(u)
                            used_in_matching.add(w)
                            break

                for u, w in matched:
                    triangles.append((v, u, w))
                    key = (min(u, w), max(u, w))
                    used_inner.add(key)
                    remaining[v].discard(u)
                    remaining[v].discard(w)

            return triangles

        def local_improve(triangles, rng, max_iter=500):
            """Try to swap triangles to increase count via 2-opt style moves"""
            triangles = list(triangles)
            # Index triangles by v
            by_v = defaultdict(list)
            for i, (v, u, w) in enumerate(triangles):
                by_v[v].append(i)

            inner_used = set()
            for v, u, w in triangles:
                inner_used.add((min(u, w), max(u, w)))

            best = len(triangles)
            improved = True
            iteration = 0
            while improved and iteration < max_iter:
                improved = False
                iteration += 1
                if time.time() > DEADLINE:
                    break

                # Try removing one triangle and re-packing its v
                # Or try swapping: remove two triangles, add two new ones
                # Simpler: for a random v, remove its triangles, re-pack v optimally
                # But this might not improve globally

                # Better: try to find a v where we can add one more triangle
                # by rearranging existing triangles
                for v in list(by_v.keys()):
                    if time.time() > DEADLINE:
                        break
                    v_tris = by_v[v]
                    if len(v_tris) == 0:
                        continue
                    # Current cross edges used by v's triangles
                    current_cross = set()
                    for i in v_tris:
                        _, u, w = triangles[i]
                        current_cross.add(u)
                        current_cross.add(w)
                    # Cross edges available for v
                    avail = cross_adj[v] - current_cross
                    if len(avail) < 2:
                        continue
                    # Inner edges used by v's triangles
                    v_inner = set()
                    for i in v_tris:
                        _, u, w = triangles[i]
                        v_inner.add((min(u, w), max(u, w)))
                    # Try to add one more triangle for v
                    # Need two vertices a, b from avail such that (a,b) not in inner_used
                    # And we can free up an inner edge by removing one existing triangle
                    # This is complex; skip for now
                    pass

                # Alternative: try global re-pack of a few vertices
                # Pick random subset of 3-5 vertices, remove their triangles, re-pack
                if len(I) > 0 and rng.random() < 0.3:
                    subset_size = min(len(I), rng.randint(2, 5))
                    subset = rng.sample(I, subset_size)
                    # Remove triangles for these vertices
                    to_remove = set()
                    for v in subset:
                        for i in by_v.get(v, []):
                            to_remove.add(i)
                    if not to_remove:
                        continue
                    removed = []
                    for i in sorted(to_remove, reverse=True):
                        removed.append(triangles[i])
                        del triangles[i]
                    # Rebuild by_v
                    by_v = defaultdict(list)
                    for i, (v, u, w) in enumerate(triangles):
                        by_v[v].append(i)
                    inner_used = set()
                    for v, u, w in triangles:
                        inner_used.add((min(u, w), max(u, w)))
                    # Re-pack subset vertices
                    for v in subset:
                        S = set(cross_adj[v])
                        # Remove already used cross edges (from other triangles)
                        # Actually cross edges are per-vertex, so no conflict
                        # But inner edges conflict
                        S_list = sorted(S)
                        m = len(S_list)
                        used_in_matching = set()
                        for i in range(m):
                            u = S_list[i]
                            if u in used_in_matching:
                                continue
                            for j in range(i + 1, m):
                                w = S_list[j]
                                if w in used_in_matching:
                                    continue
                                key = (min(u, w), max(u, w))
                                if key not in inner_used:
                                    triangles.append((v, u, w))
                                    inner_used.add(key)
                                    used_in_matching.add(u)
                                    used_in_matching.add(w)
                                    by_v[v].append(len(triangles) - 1)
                                    break

                    if len(triangles) > best:
                        best = len(triangles)
                        improved = True

            return triangles

        best_triangles = []
        best_count = 0

        # Multiple random restarts
        n_restarts = 20
        rng_main = random.Random(42)

        for r in range(n_restarts):
            if time.time() > DEADLINE:
                break
            order_I = list(I)
            order_C = list(C)
            rng_restart = random.Random(r * 1000 + 7)
            rng_restart.shuffle(order_I)
            rng_restart.shuffle(order_C)

            tris = greedy_solve(order_I, order_C, rng_restart)
            tris = local_improve(tris, rng_restart, max_iter=200)
            if len(tris) > best_count:
                best_count = len(tris)
                best_triangles = tris[:]

        # Also try sorted order (deterministic baseline)
        order_I = list(I)
        order_C = list(C)
        tris = greedy_solve(order_I, order_C, random.Random(0))
        if len(tris) > best_count:
            best_count = len(tris)
            best_triangles = tris[:]

        # Build clique partition
        # Triangles: (v, u, w)
        cliques = []
        # Cross edges covered by triangles
        cross_in_tri = set()
        inner_in_tri = set()
        for v, u, w in best_triangles:
            cliques.append([v, u, w])
            cross_in_tri.add((min(v, u), max(v, u)))
            cross_in_tri.add((min(v, w), max(v, w)))
            inner_in_tri.add((min(u, w), max(u, w)))

        # Remaining cross edges -> pairs
        for u, v in edges:
            if min(u, v) < k and max(u, v) >= k:
                # cross edge
                key = (u, v)
                if key not in cross_in_tri:
                    cliques.append([u, v])

        # Remaining inner edges -> pairs
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in inner_in_tri:
                    cliques.append([u, v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE - 5:
            # Fallback: seed behavior (all edges as cliques)
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        cliques = solve_instance(*spec)
        results.append(cliques)

    # Ensure types are correct
    final_result = []
    for cliques in results:
        new_cliques = []
        for c in cliques:
            new_c = [int(x) for x in c]
            new_cliques.append(new_c)
        final_result.append(new_cliques)

    return final_result
# EVOLVE-BLOCK-END
