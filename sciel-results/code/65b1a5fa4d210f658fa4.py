"""Clique partition of split graphs with improved greedy and coloring."""
import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Build neighbor sets
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        # Try multiple strategies and keep best
        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Process I-vertices in decreasing degree order
        I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)

        for strategy_idx, I_order in enumerate([I_verts, list(reversed(I_verts)), sorted(range(k, n))]):
            if _time_up():
                break

            used_cc = set()
            cliques = []

            for v in I_order:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue

                # Greedy packing: try to form largest possible cliques
                remaining = list(neigh)
                while remaining:
                    # Start with first element
                    group = [remaining.pop(0)]

                    # Try to add more elements that don't conflict
                    i = 0
                    while i < len(remaining):
                        cand = remaining[i]
                        ok = True
                        for g in group:
                            e = (cand, g) if cand < g else (g, cand)
                            if e in used_cc:
                                ok = False
                                break
                        if ok:
                            group.append(cand)
                            remaining.pop(i)
                        else:
                            i += 1

                    if len(group) == 1:
                        cliques.append([v, group[0]])
                    else:
                        cliques.append([v] + group)
                        for i in range(len(group)):
                            for j in range(i + 1, len(group)):
                                a, b = group[i], group[j]
                                e = (a, b) if a < b else (b, a)
                                used_cc.add(e)

            # Cover remaining C-C edges via coloring
            H_adj = [set() for _ in range(k)]
            for (a, b) in used_cc:
                H_adj[a].add(b)
                H_adj[b].add(a)

            uncolored = set(range(k))
            while uncolored:
                # Pick vertex with max degree in remaining subgraph
                best_v = None
                best_deg = -1
                for v in uncolored:
                    deg = len(H_adj[v] & uncolored)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v

                color_class = [best_v]
                uncolored.discard(best_v)

                # Add compatible vertices
                for u in list(uncolored):
                    if u not in uncolored:
                        continue
                    compatible = all(u not in H_adj[c] for c in color_class)
                    if compatible:
                        color_class.append(u)
                        uncolored.discard(u)

                if len(color_class) >= 2:
                    cliques.append(sorted(color_class))

            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        if best_cliques is None:
            # Fallback
            best_cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    best_cliques.append([u, v])

        return best_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        cliques = _solve_one(n, k, p, seed)
        results.append(cliques)

    # Type conversion
    for i in range(len(results)):
        new_list = []
        for clique in results[i]:
            new_clique = [int(x) for x in clique]
            new_list.append(new_clique)
        results[i] = new_list

    return results
