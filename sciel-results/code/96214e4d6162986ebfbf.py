"""Clique partition of split graphs using optimized partitioning strategy."""
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
        edges = instance(n, k, p, seed)

        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh

        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        total_cc = len(all_cc_edges)

        # Precompute C-C edge pairs for each possible subset size
        def _cc_edges_in_subset(subset):
            """Return set of C-C edges within subset."""
            edges_in = set()
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    a, b = subset[i], subset[j]
                    e = (a, b) if a < b else (b, a)
                    edges_in.add(e)
            return edges_in

        def _cover_remaining_cc(used_cc, cliques_out):
            """Cover remaining C-C edges with C-only cliques via greedy coloring on complement."""
            remaining = all_cc_edges - used_cc
            if not remaining:
                return 0
            active = set()
            for (a, b) in remaining:
                active.add(a)
                active.add(b)
            if not active:
                return 0
            if len(remaining) == total_cc:
                cliques_out.append(list(range(k)))
                return 1
            n_active = len(active)
            if len(remaining) == n_active * (n_active - 1) // 2:
                cliques_out.append(sorted(active))
                return 1
            # Complement graph: edges where C-C edge is consumed
            comp_adj = [set() for _ in range(k)]
            for (a, b) in used_cc:
                comp_adj[a].add(b)
                comp_adj[b].add(a)
            # Greedy coloring on complement restricted to active vertices
            colors = {}
            uncolored = set(active)
            color_count = 0
            while uncolored:
                # Find vertex with most neighbors in complement among uncolored
                best_v = None
                best_deg = -1
                for v in uncolored:
                    deg = len(comp_adj[v] & uncolored)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                # Build color class: greedily add vertices non-adjacent in complement
                color_class = [best_v]
                candidates = set(uncolored - {best_v})
                for v in sorted(candidates, key=lambda x: len(comp_adj[x] & candidates), reverse=True):
                    if v not in candidates:
                        continue
                    if all(v not in comp_adj[c] for c in color_class):
                        color_class.append(v)
                        candidates.discard(v)
                if len(color_class) >= 2:
                    cliques_out.append(list(color_class))
                    color_count += 1
                uncolored -= set(color_class)
            return color_count

        def _try_strategy(mode, order_desc=True):
            """Try a partitioning strategy. Returns (cliques, count) or None if invalid."""
            used_cc = set()
            cliques = []
            I_verts = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=order_desc)
            for v in I_verts:
                neigh = list(N_C[v])
                d = len(neigh)
                if d == 0:
                    continue
                elif d == 1:
                    cliques.append([v, neigh[0]])
                else:
                    if mode == 'big':
                        cc_in_neigh = _cc_edges_in_subset(neigh)
                        if cc_in_neigh & used_cc:
                            for u in neigh:
                                cliques.append([v, u])
                        else:
                            cliques.append([v] + neigh)
                            used_cc |= cc_in_neigh
                    elif mode == 'greedy_pack':
                        remaining_neigh = list(neigh)
                        groups = []
                        while remaining_neigh:
                            group = [remaining_neigh.pop(0)]
                            changed = True
                            while changed:
                                changed = False
                                for idx in range(len(remaining_neigh)):
                                    cand = remaining_neigh[idx]
                                    ok = all(
                                        (cand, g) if cand < g else (g, cand) not in used_cc
                                        for g in group
                                    )
                                    if ok:
                                        group.append(cand)
                                        remaining_neigh.pop(idx)
                                        changed = True
                                        break
                            groups.append(group)
                        for group in groups:
                            if len(group) == 1:
                                cliques.append([v, group[0]])
                            else:
                                cliques.append([v] + group)
                                used_cc |= _cc_edges_in_subset(group)
                    elif mode == 'individual':
                        for u in neigh:
                            cliques.append([v, u])
                    elif mode == 'big_then_split':
                        cc_in_neigh = _cc_edges_in_subset(neigh)
                        if not (cc_in_neigh & used_cc):
                            cliques.append([v] + neigh)
                            used_cc |= cc_in_neigh
                        else:
                            mid = d // 2
                            g1 = neigh[:mid]
                            g2 = neigh[mid:]
                            cc1 = _cc_edges_in_subset(g1)
                            if not (cc1 & used_cc) and len(g1) >= 2:
                                cliques.append([v] + g1)
                                used_cc |= cc1
                            elif len(g1) >= 2:
                                for u in g1:
                                    cliques.append([v, u])
                            elif len(g1) == 1:
                                cliques.append([v, g1[0]])
                            cc2 = _cc_edges_in_subset(g2)
                            if not (cc2 & used_cc) and len(g2) >= 2:
                                cliques.append([v] + g2)
                                used_cc |= cc2
                            elif len(g2) >= 2:
                                for u in g2:
                                    cliques.append([v, u])
                            elif len(g2) == 1:
                                cliques.append([v, g2[0]])
            ccount = _cover_remaining_cc(used_cc, cliques)
            return cliques, len(cliques)

        best_cliques = None
        best_count = float('inf')

        for mode in ['greedy_pack', 'big', 'big_then_split', 'individual']:
            if _time_up():
                break
            for order_desc in [True, False]:
                if _time_up():
                    break
                result = _try_strategy(mode, order_desc)
                if result is not None:
                    cliques, cnt = result
                    if cnt < best_count:
                        best_count = cnt
                        best_cliques = cliques

        if best_cliques is None:
            best_cliques = [list(range(k))]
            for (u, v) in edge_set:
                if u >= k or v >= k:
                    best_cliques.append([u, v])
            best_count = len(best_cliques)

        results.append(best_cliques)

    return results
