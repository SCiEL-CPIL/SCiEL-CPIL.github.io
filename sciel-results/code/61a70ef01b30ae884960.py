import time
from random import Random
import sys

def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def _validate_cliques(cliques, edge_set, n):
        """Validate: each clique is a clique, every edge in exactly one clique."""
        for cl in cliques:
            if len(cl) < 2:
                return False
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return False
        edge_cover = {}
        for cl in cliques:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    edge_cover[(a, b)] = edge_cover.get((a, b), 0) + 1
        for e in edge_set:
            if edge_cover.get(e, 0) != 1:
                return False
        return True

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

        # Greedy set cover with cost-awareness: prefer vertices that cover many C-C edges
        # but also consider cross-edge savings (fewer 2-cliques needed)
        uncovered = set(all_cc_edges)
        selected_v = []
        selected_set = set()
        while uncovered and not _time_up():
            best_v = None
            best_score = 0
            for v in range(k, n):
                if v in selected_set:
                    continue
                cc_cnt = len(uncovered & v_cover[v])
                if cc_cnt == 0:
                    continue
                # Cross edges this vertex would "absorb" (reducing 2-cliques)
                cross_cnt = len(N_C[v])
                # Net benefit: covers cc_cnt C-C edges (saves cc_cnt 2-cliques)
                # but the clique for v already exists if cross_cnt > 0
                # If cross_cnt == 0, we add a new clique just for C-C edges
                # Score: prioritize more C-C coverage, slight bonus for cross absorption
                score = cc_cnt * 2 + cross_cnt  # weight C-C coverage more
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v is None or best_score == 0:
                break
            selected_v.append(best_v)
            selected_set.add(best_v)
            uncovered -= v_cover[best_v]

        if uncovered:
            # Fall back: can't cover all C-C edges with I-vertices, use base
            return best_cliques, best_count

        # Initialize S_v: each selected I-vertex must include its C-neighbors
        # to cover cross edges, and C-C edges among those neighbors
        S_v = {v: set() for v in selected_v}
        for v in selected_v:
            S_v[v] = set(N_C[v])  # must include all C-neighbors for cross edges

        # Now assign remaining C-C edges to selected vertices
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
            # Find a selected vertex that contains both u and w
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

        if not _validate_cliques(cliques, edge_set, n):
            return best_cliques, best_count

        if len(cliques) < best_count:
            best_cliques = cliques
            best_count = len(cliques)

        # LOCAL SEARCH: try to remove I-vertices by redistributing their C-neighbors
        import random as _rnd
        _rng = _rnd.Random(seed + 999)

        for iteration in range(500):
            if _time_up():
                break
            if len(selected_v) < 2:
                break

            v_idx = _rng.randrange(len(selected_v))
            v_remove = selected_v[v_idx]
            movable = list(S_v[v_remove])
            if not movable:
                continue

            # Try to remove v_remove: check if all its C-C edges can be covered by others
            can_remove = True
            for i in range(len(movable)):
                for j in range(i + 1, len(movable)):
                    u, w = movable[i], movable[j]
                    covered = False
                    for v2 in selected_v:
                        if v2 == v_remove:
                            continue
                        if u in S_v[v2] and w in S_v[v2]:
                            covered = True
                            break
                    if not covered:
                        # Check if we can add u,w to some v2 that has both in N_C
                        for v2 in selected_v:
                            if v2 == v_remove:
                                continue
                            if u in N_C[v2] and w in N_C[v2]:
                                # Check all other pairs in S_v[v_remove] involving u or w
                                # can also be covered
                                can_add = True
                                for x in movable:
                                    if x == u or x == w:
                                        continue
                                    # edge (u,x) or (w,x) must be covered
                                    if u in N_C[v2] and x in N_C[v2]:
                                        pass  # would be covered if we add
                                    elif w in N_C[v2] and x in N_C[v2]:
                                        pass
                                    else:
                                        # This edge won't be covered by v2
                                        # Check if already covered by another
                                        already = False
                                        for v3 in selected_v:
                                            if v3 == v_remove and v3 != v2:
                                                continue
                                            if u in S_v[v3] and x in S_v[v3]:
                                                already = True
                                                break
                                            if w in S_v[v3] and x in S_v[v3]:
                                                already = True
                                                break
                                        if not already:
                                            can_add = False
                                            break
                                if can_add:
                                    covered = True
                                    break
                    if not covered:
                        can_remove = False
                        break
                if not can_remove:
                    break

            if can_remove:
                # Remove v_remove and reassign
                new_selected = [v for v in selected_v if v != v_remove]
                S_v_new = {v: set(S_v[v]) for v in new_selected}

                # Redistribute movable C-vertices
                for u in movable:
                    # Add u to a vertex that has u in N_C and can accommodate
                    placed = False
                    for v2 in new_selected:
                        if u in N_C[v2]:
                            S_v_new[v2].add(u)
                            placed = True
                            break
                    if not placed:
                        can_remove = False
                        break

                if can_remove:
                    # Verify all C-C edges covered
                    cc_new = set()
                    for v in new_selected:
                        nc_v = sorted(S_v_new[v])
                        for i in range(len(nc_v)):
                            for j in range(i + 1, len(nc_v)):
                                cc_new.add((nc_v[i], nc_v[j]))
                    if set(all_cc_edges) <= cc_new:
                        cliques_new = []
                        for v in new_selected:
                            if S_v_new[v]:
                                cliques_new.append(sorted([v] + list(S_v_new[v])))
                        if _validate_cliques(cliques_new, edge_set, n):
                            if len(cliques_new) < best_count:
                                selected_v = new_selected
                                S_v = S_v_new
                                best_cliques = cliques_new
                                best_count = len(cliques_new)

        # FINAL PASS: try to merge cliques or reduce further
        if not _time_up() and len(selected_v) >= 2:
            # Try pairwise: can we remove one and merge its neighbors into another?
            for i in range(len(selected_v)):
                if _time_up():
                    break
                vi = selected_v[i]
                for j in range(len(selected_v)):
                    if _time_up():
                        break
                    if j == i:
                        continue
                    vj = selected_v[j]
                    # Try moving all of vi's C-neighbors to vj
                    if all(u in N_C[vj] for u in S_v[vi]):
                        new_S_vj = S_v[vj] | S_v[vi]
                        # Check if vj still forms a clique
                        test_clique = sorted([vj] + list(new_S_vj))
                        valid = True
                        for a in range(len(test_clique)):
                            for b in range(a + 1, len(test_clique)):
                                x, y = test_clique[a], test_clique[b]
                                if x > y:
                                    x, y = y, x
                                if (x, y) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Remove vi, update vj
                            new_selected = [v for idx, v in enumerate(selected_v) if idx != i]
                            S_v_new = {v: set(S_v[v]) for v in new_selected}
                            S_v_new[vj] = new_S_vj
                            cliques_new = []
                            for v in new_selected:
                                if S_v_new[v]:
                                    cliques_new.append(sorted([v] + list(S_v_new[v])))
                            if _validate_cliques(cliques_new, edge_set, n):
                                if len(cliques_new) < best_count:
                                    selected_v = new_selected
                                    S_v = S_v_new
                                    best_cliques = cliques_new
                                    best_count = len(cliques_new)
                                    break

        return best_cliques, best_count

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    # Allocate time proportionally: give more to larger instances
    time_per = (_DEADLINE - (time.time() - _T0)) / len(specs)
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
