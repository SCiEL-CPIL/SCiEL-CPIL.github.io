import time
import math
import random as _random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    sys.setrecursionlimit(10000)

    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        cliques = []

        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return

            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u

            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])

            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = bron_kerbosch(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # Build graph H: union of all maximal cliques (edges between co-occurring vertices)
        # A vertex cover of H = hitting set for all maximal cliques of size >= 2
        H_adj = [set() for _ in range(n)]
        for c in cliques:
            cl = list(c)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    H_adj[cl[i]].add(cl[j])
                    H_adj[cl[j]].add(cl[i])

        # Vertices that are in at least one edge of H
        active_vertices = [v for v in range(n) if H_adj[v]]
        if not active_vertices:
            return []

        # Map to compact indices
        vmap = {v: i for i, v in enumerate(active_vertices)}
        nv = len(active_vertices)
        H_compact = [set() for _ in range(nv)]
        for v in active_vertices:
            hi = vmap[v]
            for w in H_adj[v]:
                H_compact[hi].add(vmap[w])

        # Greedy vertex cover for initial incumbent
        def greedy_vc():
            remaining_edges = set()
            for u in range(nv):
                for v in H_compact[u]:
                    if u < v:
                        remaining_edges.add((u, v))
            selected = set()
            while remaining_edges:
                # Pick vertex covering most remaining edges
                best_v = -1
                best_cnt = 0
                for v in range(nv):
                    if v in selected:
                        continue
                    cnt = 0
                    for u in H_compact[v]:
                        if u < v:
                            if (u, v) in remaining_edges:
                                cnt += 1
                        else:
                            if (v, u) in remaining_edges:
                                cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
                if best_v == -1 or best_cnt == 0:
                    break
                selected.add(best_v)
                for u in H_compact[best_v]:
                    if best_v < u:
                        remaining_edges.discard((best_v, u))
                    else:
                        remaining_edges.discard((u, best_v))
            return selected

        best_sel = greedy_vc()
        best_size = len(best_sel)

        # Maximum matching for lower bound on remaining subgraph
        def max_matching(subset):
            """Maximum matching in H induced by subset using augmenting paths."""
            match = dict()
            def try_aug(u, visited):
                for v in H_compact[u]:
                    if v not in subset or v in visited:
                        continue
                    visited.add(v)
                    if v not in match or try_aug(match[v], visited):
                        match[u] = v
                        match[v] = u
                        return True
                return False
            count = 0
            for u in subset:
                if u not in match:
                    if try_aug(u, set()):
                        count += 1
            return count

        def vc_lb(subset):
            """Lower bound: max matching size in induced subgraph."""
            if not subset:
                return 0
            return max_matching(subset)

        # Branch and bound for minimum vertex cover
        def vc_bnb(selected, unselected):
            nonlocal best_sel, best_size
            if time.time() - start_time > DEADLINE - 5:
                return
            # Check if all edges are covered
            uncovered = set()
            for u in unselected:
                for v in H_compact[u]:
                    if v in unselected and u < v:
                        uncovered.add((u, v))
            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = set(selected)
                return
            if len(selected) >= best_size:
                return
            # Lower bound: matching in unselected subgraph
            lb = len(selected) + vc_lb(unselected)
            if lb >= best_size:
                return
            # Pick an uncovered edge to branch on
            edge = next(iter(uncovered))
            u, v = edge
            # Branch 1: select u
            vc_bnb(selected | {u}, unselected - {u})
            # Branch 2: select v
            vc_bnb(selected | {v}, unselected - {v})

        all_unselected = set(range(nv))
        vc_bnb(set(), all_unselected)

        # Local search polish
        def is_vc(sel):
            for u in sel:
                pass
            # Check every edge has at least one endpoint in sel
            for u in range(nv):
                for v in H_compact[u]:
                    if u < v and u not in sel and v not in sel:
                        return False
            return True

        def local_search_vc(sel):
            sel = set(sel)
            improved = True
            while improved:
                improved = False
                # Try removing each vertex
                for v in list(sel):
                    trial = sel - {v}
                    if is_vc(trial):
                        sel = trial
                        improved = True
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Try 1-for-1 swaps
                for v in list(sel):
                    for w in range(nv):
                        if w in sel:
                            continue
                        trial = (sel - {v}) | {w}
                        if is_vc(trial) and len(trial) < len(sel):
                            sel = trial
                            improved = True
                            break
                    if improved:
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
            return sel

        best_sel = local_search_vc(best_sel)
        best_size = len(best_sel)

        # Additional polish with perturbations
        if time.time() - start_time < DEADLINE - 15:
            rng = _random.Random(42)
            for _ in range(20):
                if time.time() - start_time > DEADLINE - 10:
                    break
                perturbed = set(best_sel)
                if not perturbed:
                    break
                remove_idx = rng.choice(list(perturbed))
                perturbed.discard(remove_idx)
                # Restore coverage
                while not is_vc(perturbed):
                    # Find uncovered edge
                    unc_edge = None
                    for u in range(nv):
                        for v in H_compact[u]:
                            if u < v and u not in perturbed and v not in perturbed:
                                unc_edge = (u, v)
                                break
                        if unc_edge:
                            break
                    if not unc_edge:
                        break
                    # Add a random endpoint
                    u, v = unc_edge
                    if rng.random() < 0.5:
                        perturbed.add(u)
                    else:
                        perturbed.add(v)
                polished = local_search_vc(perturbed)
                if len(polished) < best_size:
                    best_size = len(polished)
                    best_sel = polished

        best_vertices = sorted(active_vertices[i] for i in best_sel)
        return best_vertices

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
