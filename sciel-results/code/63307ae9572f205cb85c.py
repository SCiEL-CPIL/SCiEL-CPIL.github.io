import time
from random import Random
import sys

import heapq
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

    def _verify_and_count(cliques, edge_set, n, k):
        """Verify cliques form a valid partition; return count or -1 if invalid."""
        edge_cover = {}
        for cl in cliques:
            if len(cl) < 2:
                return -1
            # Check all pairs are edges
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return -1
                    edge_cover[(a, b)] = edge_cover.get((a, b), 0) + 1
        # Every edge covered exactly once
        for e in edge_set:
            if edge_cover.get(e, 0) != 1:
                return -1
        return len(cliques)

    def _build_cliques(selected_v, S_v):
        """Build clique list from selected I-vertices and their C-neighbor assignments."""
        cliques = []
        for v in selected_v:
            if S_v[v]:
                clique = sorted([v] + list(S_v[v]))
                cliques.append(clique)
        return cliques

    def _try_remove_v(selected_v, S_v, v_remove, N_C, cross_edges_norm, all_cc_edges, edge_set):
        """Try to remove v_remove by redistributing its C-vertices to others.
        Returns (new_selected_v, new_S_v) or (None, None) if impossible."""
        if len(selected_v) < 2:
            return None, None
        
        remaining = [v for v in selected_v if v != v_remove]
        
        # For each C-vertex in S_v[v_remove], find a target
        # that can cover all required C-C edges
        new_S = {v: set() for v in remaining}
        
        # Start with cross-edge requirements
        for (u, v) in cross_edges_norm:
            if v in new_S:
                new_S[v].add(u)
        
        # Try to assign each C-vertex from v_remove
        movable = list(S_v[v_remove])
        for u in movable:
            assigned = False
            for v2 in remaining:
                if u not in N_C[v2]:
                    continue
                # Check if adding u to v2 keeps all C-C edges covered
                # All C-C edges involving u and other C-vertices in S_v[v_remove]
                # must be covered by some clique
                ok = True
                for w in movable:
                    if w == u:
                        continue
                    # Edge (u, w) must be in some clique
                    # It's covered if there exists v3 (including v2) where both u and w are in S[v3]
                    covered = False
                    for v3 in remaining:
                        if u in new_S[v3] and w in new_S[v3]:
                            covered = True
                            break
                        if v3 == v2 and u == w:  # same vertex, skip
                            covered = True
                            break
                    if not covered:
                        # Check if v2 can cover it
                        if u in N_C[v2] and w in N_C[v2]:
                            # Would be covered if we add u to v2
                            # But we need w also in v2's set
                            if w in new_S[v2] or w in movable:
                                # w is in v_remove's set, will be handled
                                # Actually we need both in same clique
                                pass
                    # Simpler: just check if after adding u to v2, all C-C edges
                    # between u and other movable vertices are covered
                    for w in movable:
                        if w == u:
                            continue
                        if (min(u,w), max(u,w)) not in [(min(a,b),max(a,b)) for a in new_S[v2] for b in new_S[v2] if a < b]:
                            # This edge won't be in v2's clique
                            # Check other cliques
                            found = False
                            for v3 in remaining:
                                if v3 != v2:
                                    if u in new_S[v3] and w in new_S[v3]:
                                        found = True
                                        break
                            if not found:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    new_S[v2].add(u)
                    assigned = True
                    break
            if not assigned:
                return None, None
        
        # Now assign remaining C-C edges not covered
        cc_covered = set()
        for v in remaining:
            nc_v = sorted(new_S[v])
            for i in range(len(nc_v)):
                for j in range(i + 1, len(nc_v)):
                    cc_covered.add((nc_v[i], nc_v[j]))
        
        remaining_cc = set(all_cc_edges) - cc_covered
        for (u, w) in sorted(remaining_cc):
            assigned = False
            for v in remaining:
                if u in N_C[v] and w in N_C[v]:
                    new_S[v].add(u)
                    new_S[v].add(w)
                    assigned = True
                    break
            if not assigned:
                return None, None
        
        # Verify
        cliques = _build_cliques(remaining, new_S)
        cnt = _verify_and_count(cliques, edge_set, n, k)
        if cnt < 0:
            return None, None
        
        return remaining, new_S

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
        if not all_cc_edges:
            return best_cliques, best_count

        # Precompute which C-C edges each I-vertex can cover
        v_cover = {}
        for v in range(k, n):
            nc = sorted(N_C[v])
            if len(nc) >= 2:
                v_cover[v] = frozenset((nc[i], nc[j]) for i in range(len(nc)) for j in range(i + 1, len(nc)))
            else:
                v_cover[v] = frozenset()

        # Improved greedy set cover with priority queue
        uncovered = set(all_cc_edges)
        selected_v = []
        selected_set = set()
        
        # Use a heap: (-count, v) for greedy
        # We'll recompute counts as we go
        while uncovered and not _time_up():
            best_v = None
            best_cnt = 0
            for v in range(k, n):
                if v in selected_set:
                    continue
                cnt = len(uncovered & v_cover[v])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None or best_cnt == 0:
                break
            selected_v.append(best_v)
            selected_set.add(best_v)
            uncovered -= v_cover[best_v]

        if uncovered:
            return best_cliques, best_count

        # Initialize S_v with cross-edge requirements
        S_v = {v: set() for v in range(k, n)}
        for (u, v) in cross_edges_norm:
            if v in selected_set:
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

        # Build initial cliques and verify
        cliques = _build_cliques(selected_v, S_v)
        cnt = _verify_and_count(cliques, edge_set, n, k)
        if cnt >= 0:
            best_cliques = cliques
            best_count = cnt
        else:
            return best_cliques, best_count

        # LOCAL SEARCH: try removing vertices
        import random as _rnd
        _rng = _rnd.Random(seed + 999)
        
        for iteration in range(500):
            if _time_up():
                break
            if len(selected_v) < 2:
                break

            v_idx = _rng.randrange(len(selected_v))
            v_remove = selected_v[v_idx]
            
            new_selected, new_S = _try_remove_v(
                selected_v, S_v, v_remove, N_C, cross_edges_norm, all_cc_edges, edge_set
            )
            
            if new_selected is not None:
                selected_v = new_selected
                S_v = new_S
                cliques = _build_cliques(selected_v, S_v)
                cnt = _verify_and_count(cliques, edge_set, n, k)
                if cnt >= 0 and cnt < best_count:
                    best_cliques = cliques
                    best_count = cnt

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
