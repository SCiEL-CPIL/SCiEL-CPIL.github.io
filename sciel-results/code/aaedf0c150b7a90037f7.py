"""Clique partition of split graphs via greedy absorption + Max-Clique cover."""
import time
from random import Random


def construct():
    _T0 = time.time()
    _DEADLINE = 135.0  # safety margin before 150s kill

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

    def _cover_remaining_cc(remaining, k):
        """Cover the remaining C-C edges with C-only cliques.
        Uses Maximum Clique extraction.
        """
        if not remaining:
            return []
        
        cliques = []
        remaining = set(remaining)
        
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

    def _is_clique(vertices, edge_set):
        """Check if all pairs in vertices form an edge."""
        verts = list(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if _cc_edge(verts[i], verts[j]) not in edge_set:
                    return False
        return True

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
        cc_list = list(all_cc)

        # For each C-C edge (u,v), find I-vertices adjacent to both u and v
        cc_to_iverts = {}
        for idx, (u, v) in enumerate(cc_list):
            candidates = []
            for w in range(k, n):
                if u in N_C[w] and v in N_C[w]:
                    candidates.append(w)
            cc_to_iverts[idx] = candidates

        # Phase 1: Absorb I-vertices whose C-neighborhood is a clique into large cliques
        # This covers many cross-edges and C-C edges simultaneously
        used_cross = set()  # cross edges already covered
        used_cc = set()     # C-C edges already covered
        cliques = []
        absorbed_i_verts = set()
        
        # Sort I-vertices by size of their C-neighborhood (larger first for better absorption)
        i_vert_order = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
        
        for w in i_vert_order:
            if _time_up():
                break
            nc = N_C[w]
            if len(nc) < 2:
                continue
            
            # Check if nc forms a clique
            if _is_clique(nc, edge_set):
                # Create clique: nc ∪ {w}
                clique = sorted(list(nc) + [w])
                
                # Check if all C-C edges within nc are still uncovered
                all_cc_in_nc = True
                for i in range(len(nc)):
                    for j in range(i + 1, len(nc)):
                        if _cc_edge(nc[i], nc[j]) in used_cc:
                            all_cc_in_nc = False
                            break
                    if not all_cc_in_nc:
                        break
                
                # Check if all cross edges from nc to w are still uncovered
                all_cross_ok = True
                for u in nc:
                    if _cc_edge(u, w) in used_cross:
                        all_cross_ok = False
                        break
                
                if all_cc_in_nc and all_cross_ok:
                    cliques.append(clique)
                    # Mark C-C edges within nc as used
                    for i in range(len(nc)):
                        for j in range(i + 1, len(nc)):
                            used_cc.add(_cc_edge(nc[i], nc[j]))
                    # Mark cross edges from nc to w as used
                    for u in nc:
                        used_cross.add(_cc_edge(u, w))
                    absorbed_i_verts.add(w)
        
        # Phase 2: Cover remaining cross-edges with 2-cliques or triangles
        # For remaining I-vertices, try to form triangles with C-C edges
        remaining_i_verts = [w for w in range(k, n) if w not in absorbed_i_verts]
        
        # Build list of remaining C-C edges
        remaining_cc_list = [e for e in cc_list if e not in used_cc]
        
        # For each remaining C-C edge, find I-vertices that can form a triangle
        cc_to_iverts_remaining = {}
        for idx, (u, v) in enumerate(remaining_cc_list):
            candidates = []
            for w in remaining_i_verts:
                if u in N_C[w] and v in N_C[w]:
                    # Check if cross edges (u,w) and (v,w) are not yet covered
                    if _cc_edge(u, w) not in used_cross and _cc_edge(v, w) not in used_cross:
                        candidates.append(w)
            cc_to_iverts_remaining[idx] = candidates
        
        # Greedy: process C-C edges in order of most constrained first
        cc_order = sorted(range(len(remaining_cc_list)), key=lambda i: len(cc_to_iverts_remaining[i]))
        
        for idx in cc_order:
            if _time_up():
                break
            u, v = remaining_cc_list[idx]
            if (u, v) in used_cc:
                continue
            for w in cc_to_iverts_remaining[idx]:
                if _cc_edge(u, w) not in used_cross and _cc_edge(v, w) not in used_cross:
                    cliques.append([u, v, w])
                    used_cc.add((u, v))
                    used_cross.add(_cc_edge(u, w))
                    used_cross.add(_cc_edge(v, w))
                    break
        
        # Phase 3: Cover remaining cross-edges with 2-cliques
        for w in remaining_i_verts:
            for u in sorted(N_C[w]):
                if _cc_edge(u, w) not in used_cross:
                    cliques.append([u, w])
                    used_cross.add(_cc_edge(u, w))
        
        # Phase 4: Cover remaining C-C edges with C-only cliques
        remaining_cc = all_cc - used_cc
        c_only = _cover_remaining_cc(remaining_cc, k)
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
