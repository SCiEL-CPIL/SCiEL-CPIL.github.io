"""Clique partition of split graphs via greedy set cover + Max-Clique cover."""
import time
from random import Random


def construct():
    _T0 = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _cc_edge(a, b):
        return (a, b) if a < b else (b, a)

    def _max_clique(adj, n):
        """Find a maximum clique in a graph with n vertices and adjacency list adj.
        Uses Bron-Kerbosch with pivot.
        Returns the clique as a set of vertices.
        """
        best_clique = set()
        
        def bronk(R, P, X):
            nonlocal best_clique
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            
            # Choose pivot u from P U X
            pivot = None
            max_deg = -1
            for u in P:
                deg = len(adj[u] & P)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if not P:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return

            # Iterate over P \ N(pivot)
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                R.add(v)
                bronk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)
                
                # Prune if can't beat best
                if len(R) + len(P) <= len(best_clique):
                    break

        bronk(set(), set(range(n)), set())
        return best_clique

    def _cover_cc_edges(remaining_cc, k):
        """Cover C-C edges with maximum cliques in the C-subgraph."""
        if not remaining_cc:
            return []
        
        cliques = []
        remaining = set(remaining_cc)
        
        while remaining:
            # Build adjacency for current remaining edges
            adj = [set() for _ in range(k)]
            active = set()
            for (a, b) in remaining:
                adj[a].add(b)
                adj[b].add(a)
                active.add(a)
                active.add(b)
            
            if not active:
                break

            active_list = list(active)
            idx_map = {v: i for i, v in enumerate(active_list)}
            m = len(active_list)
            local_adj = [set() for _ in range(m)]
            
            for (a, b) in remaining:
                if a in idx_map and b in idx_map:
                    ia = idx_map[a]
                    ib = idx_map[b]
                    local_adj[ia].add(ib)
                    local_adj[ib].add(ia)
            
            max_cl = _max_clique(local_adj, m)
            
            if not max_cl:
                break
            
            clique_vertices = {active_list[i] for i in max_cl}
            
            cl_sorted = sorted(clique_vertices)
            for i in range(len(cl_sorted)):
                for j in range(i + 1, len(cl_sorted)):
                    remaining.discard(_cc_edge(cl_sorted[i], cl_sorted[j]))
            
            cliques.append(cl_sorted)
            
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_cc_edge(a, b))

        # N_C[v] = neighbors of I-vertex v in C
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if _cc_edge(u, v) in edge_set)

        # all C-C edges
        all_cc = set(_cc_edge(u, v) for u in range(k) for v in range(u + 1, k))
        # all C-I edges
        all_ci = set(_cc_edge(u, v) for u in range(k) for v in range(k, n) if _cc_edge(u, v) in edge_set)

        # For each I-vertex w with |N_C(w)| >= 2, candidate clique is {w} ∪ N_C(w)
        # This covers: all C-I edges (u,w) for u in N_C(w), and all C-C edges within N_C(w)
        
        # Greedy set cover: repeatedly pick the candidate clique that covers the most uncovered edges
        uncovered_cc = set(all_cc)
        uncovered_ci = set(all_ci)
        cliques = []
        
        # Build candidate cliques
        candidates = []
        for w in range(k, n):
            if len(N_C[w]) >= 2:
                clique = [w] + list(N_C[w])
                # Count how many uncovered edges this covers
                cc_covered = 0
                ci_covered = 0
                for i in range(len(N_C[w])):
                    for j in range(i + 1, len(N_C[w])):
                        e = _cc_edge(N_C[w][i], N_C[w][j])
                        if e in uncovered_cc:
                            cc_covered += 1
                for u in N_C[w]:
                    e = _cc_edge(u, w)
                    if e in uncovered_ci:
                        ci_covered += 1
                total_covered = cc_covered + ci_covered
                if total_covered > 0:
                    candidates.append((total_covered, w, clique))
        
        # Sort by coverage (descending)
        candidates.sort(key=lambda x: -x[0])
        
        # Greedy: pick best candidate, update uncovered edges, repeat
        while uncovered_cc or uncovered_ci:
            best = None
            best_cover = 0
            for (cover, w, clique) in candidates:
                # Recalculate coverage with current uncovered sets
                cc_covered = 0
                ci_covered = 0
                for i in range(len(N_C[w])):
                    for j in range(i + 1, len(N_C[w])):
                        e = _cc_edge(N_C[w][i], N_C[w][j])
                        if e in uncovered_cc:
                            cc_covered += 1
                for u in N_C[w]:
                    e = _cc_edge(u, w)
                    if e in uncovered_ci:
                        ci_covered += 1
                total_covered = cc_covered + ci_covered
                if total_covered > best_cover:
                    best_cover = total_covered
                    best = (w, clique)
            
            if best is None or best_cover == 0:
                break
            
            w, clique = best
            cliques.append(clique)
            
            # Remove covered edges
            for i in range(len(N_C[w])):
                for j in range(i + 1, len(N_C[w])):
                    uncovered_cc.discard(_cc_edge(N_C[w][i], N_C[w][j]))
            for u in N_C[w]:
                uncovered_ci.discard(_cc_edge(u, w))
            
            # Remove this candidate
            candidates = [(c, ww, cl) for (c, ww, cl) in candidates if ww != w]
        
        # Cover remaining C-I edges with 2-cliques
        for w in range(k, n):
            for u in sorted(N_C[w]):
                if _cc_edge(u, w) in uncovered_ci:
                    cliques.append([u, w])
                    uncovered_ci.discard(_cc_edge(u, w))
        
        # Cover remaining C-C edges with C-only cliques
        if uncovered_cc:
            c_only = _cover_cc_edges(uncovered_cc, k)
            for cl in c_only:
                cliques.append(cl)
        
        return cliques

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
        results.append(solve_one(*spec))

    # convert to plain python int lists
    out = []
    for clist in results:
        nl = []
        for c in clist:
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
