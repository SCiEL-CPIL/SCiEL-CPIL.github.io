import time
from random import Random
from itertools import combinations

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

    def min_clique_cover_dp(vertices, adj):
        """
        vertices: list of vertex indices
        adj: set of edges (frozenset or set of tuples)
        Returns: min number of cliques to cover vertices, and the cliques.
        Assumes |vertices| <= 20.
        """
        m = len(vertices)
        if m == 0:
            return 0, []
        
        # Map vertices to 0..m-1
        v_map = {v: i for i, v in enumerate(vertices)}
        
        # Build adjacency bitmask for each vertex
        # adj_mask[i] has bit j set if (vertices[i], vertices[j]) is an edge
        adj_mask = [0] * m
        for i in range(m):
            for j in range(i + 1, m):
                u, v = vertices[i], vertices[j]
                e = (u, v) if u < v else (v, u)
                if e in adj:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
        # Precompute is_clique[mask]
        # is_clique[mask] is True if the set of vertices in mask forms a clique
        is_clique = [False] * (1 << m)
        is_clique[0] = True
        for mask in range(1, 1 << m):
            # Get least significant bit
            lsb = mask & -mask
            v_idx = (lsb.bit_length() - 1)
            rest = mask ^ lsb
            # v_idx is connected to all in rest?
            # adj_mask[v_idx] & rest == rest
            if (adj_mask[v_idx] & rest) == rest:
                is_clique[mask] = True
            # else False
        
        # DP
        # dp[mask] = min cliques to cover mask
        # parent[mask] = (prev_mask, clique_mask)
        dp = [1000] * (1 << m)
        dp[0] = 0
        parent = [None] * (1 << m)
        
        for mask in range(1, 1 << m):
            # Iterate submasks of mask that are cliques
            # Optimization: only iterate maximal cliques? Or just all cliques.
            # To speed up, we can iterate submasks.
            sub = mask
            while sub:
                if is_clique[sub]:
                    prev = mask ^ sub
                    if dp[prev] + 1 < dp[mask]:
                        dp[mask] = dp[prev] + 1
                        parent[mask] = (prev, sub)
                sub = (sub - 1) & mask
                
        # Reconstruct
        cliques = []
        mask = (1 << m) - 1
        while mask:
            prev, sub = parent[mask]
            # Convert sub bitmask back to vertices
            clique_verts = [vertices[i] for i in range(m) if (sub >> i) & 1]
            cliques.append(clique_verts)
            mask = prev
            
        return dp[(1 << m) - 1], cliques

    def min_clique_cover_backtrack(vertices, adj):
        """
        Backtracking coloring of complement.
        Returns min number of cliques and the cliques.
        """
        m = len(vertices)
        if m == 0:
            return 0, []
            
        # Build complement adjacency
        # comp_adj[i] has bit j set if NOT an edge in G
        comp_adj = [0] * m
        for i in range(m):
            for j in range(i + 1, m):
                u, v = vertices[i], vertices[j]
                e = (u, v) if u < v else (v, u)
                if e not in adj:
                    comp_adj[i] |= (1 << j)
                    comp_adj[j] |= (1 << i)
                    
        # Order vertices by degree in complement (descending)
        # This helps in coloring
        order = sorted(range(m), key=lambda i: comp_adj[i].bit_count(), reverse=True)
        
        best_colors = m # Upper bound
        best_coloring = None
        
        def backtrack(idx, coloring, num_colors):
            nonlocal best_colors, best_coloring
            if num_colors >= best_colors:
                return
            
            if idx == m:
                best_colors = num_colors
                best_coloring = coloring[:]
                return
                
            v = order[idx]
            
            # Try assigning colors 0 to num_colors
            for c in range(num_colors):
                # Check if v can have color c
                # v must not be connected to any vertex with color c in complement
                # i.e., no neighbor in comp_adj has color c
                conflict = False
                for u in range(m):
                    if coloring[u] == c:
                        if (comp_adj[v] >> u) & 1:
                            conflict = True
                            break
                if not conflict:
                    coloring[v] = c
                    backtrack(idx + 1, coloring, num_colors)
                    
            # Try new color
            if num_colors + 1 < best_colors:
                coloring[v] = num_colors
                backtrack(idx + 1, coloring, num_colors + 1)

        coloring = [-1] * m
        backtrack(0, coloring, 0)
        
        # Reconstruct cliques from coloring
        cliques = []
        if best_coloring:
            for c in range(best_colors):
                clique_verts = [vertices[i] for i in range(m) if best_coloring[i] == c]
                cliques.append(clique_verts)
        else:
            # Fallback
            for v in vertices:
                cliques.append([v])
                
        return best_colors, cliques

    def get_min_clique_cover(vertices, adj):
        if len(vertices) <= 20:
            return min_clique_cover_dp(vertices, adj)
        else:
            return min_clique_cover_backtrack(vertices, adj)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            # Fallback
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
            
        C_verts = list(range(k))
        I_verts = list(range(k, n))
        
        # Precompute N_C for each v in I
        N_C = {}
        for v in I_verts:
            N_C[v] = [u for u in C_verts if (u, v) in edge_set]
            
        # Step 1: For each v in I, find min clique cover of G[N_C(v)]
        # adj for G[N_C(v)] is just the edges in edge_set restricted to N_C(v)
        all_cliques = []
        covered_edges = set()
        
        for v in I_verts:
            neigh = N_C[v]
            if not neigh:
                continue
                
            # Get edges within neigh
            # adj is global edge_set
            count, cliques = get_min_clique_cover(neigh, edge_set)
            
            for S in cliques:
                # Clique is {v} + S
                all_cliques.append([v] + S)
                # Mark internal edges covered
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        u, w = S[i], S[j]
                        e = (u, w) if u < w else (w, u)
                        covered_edges.add(e)
                        
        # Step 2: Find remaining internal edges in C
        remaining_edges = set()
        for u in C_verts:
            for w in C_verts:
                if u < w:
                    e = (u, w)
                    if e in edge_set and e not in covered_edges:
                        remaining_edges.add(e)
                        
        # Step 3: Find min clique cover for remaining edges
        # The vertices involved in remaining_edges
        rem_verts = set()
        for u, w in remaining_edges:
            rem_verts.add(u)
            rem_verts.add(w)
            
        if rem_verts:
            rem_verts_list = list(rem_verts)
            # adj is still edge_set
            count, cliques = get_min_clique_cover(rem_verts_list, edge_set)
            for S in cliques:
                all_cliques.append(S)
                
        # Convert to lists of ints
        final_cliques = [list(c) for c in all_cliques]
        results.append(final_cliques)

    return results
