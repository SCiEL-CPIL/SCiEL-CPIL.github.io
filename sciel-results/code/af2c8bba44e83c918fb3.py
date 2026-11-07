"""Clique partition of split graphs via greedy absorption + Max-Clique cover."""
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
            # Heuristic: pick u with max degree in P
            pivot = None
            max_deg = -1
            for u in P:
                deg = len(adj[u] & P)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # If P is empty, pivot can be from X or None
            if not P:
                if X:
                    # Pick any from X? Or just return if X not empty?
                    # Standard BK: if P is empty, we are done with this branch if we update best.
                    # Actually, if P is empty, no more vertices can be added.
                    # The condition `if not P and not X` handles the leaf.
                    # If P is empty but X is not, we still check best_clique.
                    if len(R) > len(best_clique):
                        best_clique = set(R)
                    return

            # Iterate over P \ N(pivot)
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                # Add v to R
                R.add(v)
                # Recurse
                bronk(R, P & adj[v], X & adj[v])
                # Remove v from R
                R.discard(v)
                # Move v from P to X
                P.discard(v)
                X.add(v)
                
                # Early termination? 
                # If len(R) + len(P) <= len(best_clique), we can prune.
                if len(R) + len(P) <= len(best_clique):
                    # We can't beat best_clique in this branch
                    # But we still need to move remaining P to X? 
                    # Actually, if we prune, we just return.
                    # But we need to restore state? 
                    # In this implementation, P and X are passed by reference (mutated).
                    # This is tricky. 
                    # Let's stick to simple recursion without aggressive pruning for n=30.
                    pass

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
        
        # We need to rebuild adjacency for the current remaining graph
        # Vertices are 0..k-1
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

            # Map active vertices to 0..m-1 for the BK algorithm?
            # Or just run BK on the full graph but restricting to active?
            # BK on full graph with P=active is fine.
            
            # Create a dummy graph structure for BK?
            # BK needs adj lists. 
            # We have adj for all k vertices.
            
            # Run Max Clique on the subgraph induced by 'active'
            # We can just call _max_clique with a wrapper or modify it.
            # Let's create a local adjacency list for active vertices only to be safe and fast.
            
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
                # Should not happen if remaining is not empty
                break
            
            # Map back to original vertex ids
            clique_vertices = {active_list[i] for i in max_cl}
            
            # Remove edges within this clique from remaining
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

        # Strategy:
        # 1. For each I-vertex w, find Max Clique in G[N(w)].
        # 2. Form candidate Type-B clique K_w = M_w U {w}.
        # 3. Greedily select K_w's that don't conflict on C-C edges.
        # 4. Cover remaining cross edges with 2-cliques.
        # 5. Cover remaining C-C edges with C-only max cliques.

        # Precompute C-adjacency for fast clique finding in subgraphs
        c_adj = [set() for _ in range(k)]
        for (u, v) in all_cc:
            c_adj[u].add(v)
            c_adj[v].add(u)

        candidates = []
        for w in range(k, n):
            neighbors = list(N_C[w])
            if len(neighbors) < 2:
                continue
            
            # Find max clique in induced subgraph of neighbors
            # Map neighbors to 0..m-1
            m = len(neighbors)
            idx_map = {v: i for i, v in enumerate(neighbors)}
            local_adj = [set() for _ in range(m)]
            for i in range(m):
                u = neighbors[i]
                for v in neighbors[i+1:]:
                    if v in c_adj[u]:
                        local_adj[i].add(idx_map[v])
                        local_adj[idx_map[v]].add(i)
            
            max_cl_idx = _max_clique(local_adj, m)
            if not max_cl_idx:
                continue
            
            clique_verts = [neighbors[i] for i in max_cl_idx]
            # The clique is clique_verts + [w]
            # Edges covered:
            # - All pairs in clique_verts (C-C edges)
            # - All pairs (u, w) for u in clique_verts (Cross edges)
            
            # Calculate benefit: number of edges covered
            # This helps prioritize large cliques
            benefit = len(clique_verts) + len(clique_verts)*(len(clique_verts)-1)//2
            
            # Store candidate
            # We need to track which C-C edges are covered to check conflicts
            covered_cc_edges = set()
            cv = sorted(clique_verts)
            for i in range(len(cv)):
                for j in range(i+1, len(cv)):
                    covered_cc_edges.add((cv[i], cv[j]))
            
            candidates.append({
                'w': w,
                'verts': clique_verts,
                'benefit': benefit,
                'cc_edges': covered_cc_edges
            })

        # Sort candidates by benefit descending
        candidates.sort(key=lambda x: x['benefit'], reverse=True)
        
        used_cc = set()
        covered_cross = set()
        cliques = []
        
        for cand in candidates:
            # Check if any C-C edge in this candidate is already used
            conflict = False
            for e in cand['cc_edges']:
                if e in used_cc:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Check if any cross edge is already covered
            # Cross edges are (u, w) for u in cand['verts']
            w = cand['w']
            for u in cand['verts']:
                if _cc_edge(u, w) in covered_cross:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Accept this candidate
            cliques.append(cand['verts'] + [w])
            
            for e in cand['cc_edges']:
                used_cc.add(e)
            
            for u in cand['verts']:
                covered_cross.add(_cc_edge(u, w))

        # Cover remaining cross-edges with 2-cliques
        for w in range(k, n):
            for u in sorted(N_C[w]):
                if _cc_edge(u, w) not in covered_cross:
                    cliques.append([u, w])
                    covered_cross.add(_cc_edge(u, w))
        
        # Cover remaining C-C edges with C-only cliques
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
            # Ensure int and list
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
