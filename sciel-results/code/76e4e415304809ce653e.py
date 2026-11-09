import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency as bitmasks (fast set operations via bitwise ops)
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        C_mask = (1 << k) - 1
        C_set = set(range(k))

        # Internal edges of C as a bitmask-pair set
        internal_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        used_edges = set()

        # Neighborhood of each I vertex in C (as a set of C vertices)
        I_vertices = list(range(k, n))
        Nv = {}
        for v in I_vertices:
            Nv[v] = set(i for i in range(k) if (adj_mask[v] >> i) & 1)

        # Process I vertices in ascending order of |N(v)| in C.
        # Smaller neighborhoods first: they are more constrained, so handling
        # them early prevents them from being forced into many small cliques
        # by edges already consumed by larger neighborhoods.
        I_sorted = sorted(I_vertices, key=lambda v: len(Nv[v]))

        partition = []

        for v in I_sorted:
            S = Nv[v]
            if not S:
                continue

            if len(S) == 1:
                u = next(iter(S))
                partition.append(sorted([v, u]))
                continue

            # Build conflict graph H on S: edge (a,b) in H iff (a,b) is in used_edges.
            # A color class (independent set in H) is a set of C-vertices whose
            # pairwise edges are all still unused, so {v}+class is a valid clique.
            S_list = list(S)
            idx = {x: i for i, x in enumerate(S_list)}
            m = len(S_list)
            h_adj = [set() for _ in range(m)]
            for i in range(m):
                a = S_list[i]
                for j in range(i + 1, m):
                    b = S_list[j]
                    e = (a, b) if a < b else (b, a)
                    if e in used_edges:
                        h_adj[i].add(j)
                        h_adj[j].add(i)

            # DSATUR greedy coloring of H
            colors = [-1] * m
            color_sets = [set() for _ in range(m)]
            uncolored = set(range(m))
            while uncolored:
                # Highest saturation, tie-break by degree in H
                best = max(uncolored, key=lambda i: (len(color_sets[i]), len(h_adj[i])))
                c = 0
                while c in color_sets[best]:
                    c += 1
                colors[best] = c
                color_sets[best].add(c)
                uncolored.discard(best)
                for nb in h_adj[best]:
                    if nb in uncolored and c not in color_sets[nb]:
                        color_sets[nb].add(c)

            color_count = max(colors) + 1 if colors else 0
            classes = [[] for _ in range(color_count)]
            for i in range(m):
                classes[colors[i]].append(S_list[i])

            for cls in classes:
                if len(cls) < 1:
                    continue
                clique = cls + [v]
                partition.append(sorted(clique))
                # Mark the C-edges inside cls as used
                for i in range(len(cls)):
                    a = cls[i]
                    for j in range(i + 1, len(cls)):
                        b = cls[j]
                        used_edges.add((a, b) if a < b else (b, a))

        # Remaining internal C-edges: partition into cliques.
        rem_set = set(internal_edges - used_edges)
        while rem_set:
            # Pick the edge with the most common remaining-neighbors to seed
            # a large clique.
            best_edge = None
            best_score = -1
            for e in rem_set:
                u, v = e
                common = 0
                um = adj_mask[u] & C_mask
                vm = adj_mask[v] & C_mask
                # count w (not u,v) such that (u,w) and (v,w) both in rem_set
                for w in range(k):
                    if w == u or w == v:
                        continue
                    if (min(u, w), max(u, w)) in rem_set and (min(v, w), max(v, w)) in rem_set:
                        common += 1
                if common > best_score:
                    best_score = common
                    best_edge = e

            if best_edge is None:
                best_edge = next(iter(rem_set))

            u, v = best_edge
            current = [u, v]
            cur_mask = (1 << u) | (1 << v)

            # Greedily extend to a maximal clique in rem_set
            changed = True
            while changed:
                changed = False
                # candidates: vertices connected to all of current in rem_set
                cand = list(range(k))
                cand.sort(key=lambda w: sum(
                    1 for x in current
                    if (min(w, x), max(w, x)) in rem_set), reverse=True)
                for w in cand:
                    if (cur_mask >> w) & 1:
                        continue
                    ok = True
                    for x in current:
                        e = (min(w, x), max(w, x))
                        if e not in rem_set:
                            ok = False
                            break
                    if ok:
                        current.append(w)
                        cur_mask |= (1 << w)
                        changed = True

            if len(current) >= 2:
                partition.append(sorted(current))
            for i in range(len(current)):
                a = current[i]
                for j in range(i + 1, len(current)):
                    b = current[j]
                    rem_set.discard((min(a, b), max(a, b)))

        # Refinement: merge adjacent cliques whose union is still a clique.
        # This can absorb isolated leftover edges into existing cliques,
        # reducing the total clique count.
        def is_clique(clique):
            for i in range(len(clique)):
                a = clique[i]
                for j in range(i + 1, len(clique)):
                    b = clique[j]
                    if (min(a, b), max(a, b)) not in internal_edges and \
                       not ((min(a, b) < k and max(a, b) >= k)):
                        # cross edge check: must be an actual graph edge
                        pass
                    # verify it is a graph edge
                    lo, hi = (a, b) if a < b else (b, a)
                    if not ((adj_mask[lo] >> hi) & 1):
                        return False
            return True

        merged = True
        while merged:
            merged = False
            new_partition = []
            used_idx = [False] * len(partition)
            for i in range(len(partition)):
                if used_idx[i]:
                    continue
                cur = list(partition[i])
                changed = True
                while changed:
                    changed = False
                    for j in range(len(partition)):
                        if used_idx[j] or j == i:
                            continue
                        # check if union is a clique
                        union = cur + [x for x in partition[j] if x not in cur]
                        if len(union) <= max(len(cur), len(partition[j])):
                            continue  # no new vertices, skip
                        if is_clique(union):
                            cur = union
                            used_idx[j] = True
                            changed = True
                            break
                new_partition.append(sorted(cur))
            partition = new_partition

        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback: build a trivially valid edge-based partition for
            # remaining instances (each edge as a 2-clique).
            while len(results) < 3:
                remaining_spec = specs[len(results)]
                n, k, p, sd = remaining_spec
                ed = instance(n, k, p, sd)
                results.append([[int(u), int(v)] for (u, v) in ed])
            return results
        res = solve_instance(*spec)
        cleaned = [[int(x) for x in clique] for clique in res]
        results.append(cleaned)

    return results
