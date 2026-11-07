import time
from random import Random
import sys

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _greedy_max_clique(adj_sets, vertices):
        """Greedy maximum clique in a graph defined by adj_sets on vertices."""
        if not vertices:
            return []
        v_list = list(vertices)
        start = max(v_list, key=lambda v: len(adj_sets[v]))
        clique = {start}
        remaining = set(v_list) - {start}
        while remaining:
            candidates = [v for v in remaining if all(c in adj_sets[v] for c in clique)]
            if not candidates:
                break
            best = max(candidates, key=lambda v: len(adj_sets[v]))
            clique.add(best)
            remaining.discard(best)
        return list(clique)

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if (u, v) in edge_set)

        cross_edges_norm = []
        for (a, b) in edge_set:
            if a < k and b >= k:
                cross_edges_norm.append((a, b))
            elif b < k and a >= k:
                cross_edges_norm.append((b, a))

        base_cliques = [list(range(k))]
        for (u, v) in cross_edges_norm:
            base_cliques.append(sorted([u, v]))
        best_cliques = base_cliques
        best_count = len(base_cliques)

        all_cc_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        total_cc = len(all_cc_edges)
        if total_cc == 0:
            return best_cliques, best_count

        # Precompute which C-C edges each I-vertex can cover
        v_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cover[v] = set((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cover[v] = set()

        # Greedy set cover
        uncovered = set(all_cc_edges)
        selected_v = []
        while uncovered and not _time_up():
            best_v = None
            best_cnt = 0
            for v in range(k, n):
                if v in selected_v:
                    continue
                cnt = len(uncovered & v_cover[v])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None or best_cnt == 0:
                break
            selected_v.append(best_v)
            uncovered -= v_cover[best_v]

        if uncovered:
            return best_cliques, best_count

        # Initialize S_v with cross-edge requirements
        S_v = {v: set() for v in range(k, n)}
        for (u, v) in cross_edges_norm:
            if v in selected_v:
                S_v[v].add(u)

        # Assign remaining C-C edges
        cc_covered = set()
        for v in selected_v:
            nc_v = sorted(S_v[v])
            for i in range(len(nc_v)):
                for j in range(i + 1, len(nc_v)):
                    cc_covered.add((nc_v[i], nc_v[j]))

        remaining_cc = set(all_cc_edges) - cc_covered
        for (u, w) in sorted(remaining_cc):
            if _time_up():
                break
            assigned = False
            for v in selected_v:
                if u in N_C[v] and w in N_C[v]:
                    S_v[v].add(u)
                    S_v[v].add(w)
                    assigned = True
                    break
            if not assigned:
                return best_cliques, best_count

        # Build initial cliques
        cliques = []
        for v in selected_v:
            if S_v[v]:
                clique = sorted([v] + list(S_v[v]))
                cliques.append(clique)

        # Verify
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return best_cliques, best_count

        edge_cover_count = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1

        for e in edge_set:
            if edge_cover_count.get(e, 0) != 1:
                return best_cliques, best_count

        if len(cliques) < best_count:
            best_cliques = cliques
            best_count = len(cliques)

        # LOCAL SEARCH with greedy clique extraction refinement
        import random as _rnd
        _rng = _rnd.Random(seed + 999)

        for iteration in range(200):
            if _time_up():
                break
            if len(selected_v) < 2:
                break

            v_idx = _rng.randrange(len(selected_v))
            v_remove = selected_v[v_idx]
            movable = list(S_v[v_remove])
            if not movable:
                continue

            # Try to redistribute v_remove's C-vertices to other selected I-vertices
            can_remove = True
            for u in movable:
                found = False
                for v2 in selected_v:
                    if v2 != v_remove and u in N_C[v2]:
                        all_covered = True
                        for w in S_v[v_remove]:
                            if w == u:
                                continue
                            covered = False
                            for v3 in selected_v:
                                if v3 != v_remove and u in S_v[v3] and w in S_v[v3]:
                                    covered = True
                                    break
                            if not covered:
                                all_covered = False
                                break
                        if all_covered:
                            found = True
                            break
                if not found:
                    can_remove = False
                    break

            if can_remove:
                selected_v.remove(v_remove)
                S_v_new = {v: set() for v in range(k, n)}
                for (u, v) in cross_edges_norm:
                    if v in selected_v:
                        S_v_new[v].add(u)

                cc_covered_new = set()
                for v in selected_v:
                    nc_v = sorted(S_v_new[v])
                    for i in range(len(nc_v)):
                        for j in range(i + 1, len(nc_v)):
                            cc_covered_new.add((nc_v[i], nc_v[j]))

                remaining_new = set(all_cc_edges) - cc_covered_new
                all_assigned = True
                for (u, w) in sorted(remaining_new):
                    assigned = False
                    for v in selected_v:
                        if u in N_C[v] and w in N_C[v]:
                            S_v_new[v].add(u)
                            S_v_new[v].add(w)
                            assigned = True
                            break
                    if not assigned:
                        all_assigned = False
                        break

                if all_assigned:
                    S_v = S_v_new
                    cliques = []
                    for v in selected_v:
                        if S_v[v]:
                            clique = sorted([v] + list(S_v[v]))
                            cliques.append(clique)

                    valid = True
                    for cl in cliques:
                        for i in range(len(cl)):
                            for j in range(i + 1, len(cl)):
                                a, b = cl[i], cl[j]
                                if a > b:
                                    a, b = b, a
                                if (a, b) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break

                    if valid:
                        edge_cover_count = {}
                        for cl in cliques:
                            for i in range(len(cl)):
                                for j in range(i + 1, len(cl)):
                                    a, b = cl[i], cl[j]
                                    if a > b:
                                        a, b = b, a
                                    edge_cover_count[(a, b)] = edge_cover_count.get((a, b), 0) + 1
                        for e in edge_set:
                            if edge_cover_count.get(e, 0) != 1:
                                valid = False
                                break

                        if valid and len(cliques) < best_count:
                            best_cliques = cliques
                            best_count = len(cliques)
                else:
                    selected_v.insert(v_idx, v_remove)

        return best_cliques, best_count

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append(sorted([u, v]))
            results.append(cliques)
            continue
        n, k, p, seed = spec
        cliques, _ = _solve_instance(n, k, p, seed)
        results.append(cliques)
    return results
