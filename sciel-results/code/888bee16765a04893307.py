def construct():
    import random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        # Get edges from the standard instance generator to ensure consistency
        edges_list = instance(n, k, p, seed)
        all_edges = set(edges_list)
        
        # Separate C-edges and I-edges
        C_edges = set()
        I_edges = set()
        N = [set() for _ in range(n)] # Neighbors in C for I vertices? No, N[v] for v in I should be neighbors in C.
        
        # Re-derive structure or just use edge sets
        for u, v in all_edges:
            if u < k and v < k:
                C_edges.add((u, v))
            elif u < k and v >= k:
                I_edges.add((u, v))
                N[v].add(u) # v is in I, u is in C
            elif u >= k and v < k:
                I_edges.add((v, u)) # Normalize to (C, I)
                N[u].add(v) # u is in I, v is in C
            # I-I edges don't exist in split graph generation, but check just in case
            # The generator only creates C-C and C-I edges.
            
        # The previous logic in solve_split was generating its own edges.
        # We need to adapt the algorithm to use these edges.
        
        # The algorithm in the failed code was:
        # 1. Handle cross edges greedily.
        # 2. Handle remaining C edges via complement coloring.
        
        # Let's reuse the algorithm logic but ensure it uses the correct edges.
        
        covered = set()
        cliques = []
        
        # Step 1: Handle cross edges
        # Identify I vertices and their neighbors in C
        I_vertices = [v for v in range(k, n)]
        # Sort by degree in C (number of neighbors in C)
        # We need N[v] for v in I.
        # From the loop above, N[v] contains neighbors in C for v in I.
        
        # Calculate degree for sorting
        deg = {v: len(N[v]) for v in I_vertices}
        I_vertices.sort(key=lambda v: deg[v], reverse=True)
        
        for v in I_vertices:
            U_v = set()
            for u in N[v]:
                # Check if edge (u, v) is uncovered
                # Edges are stored as (min, max) in sets? 
                # In all_edges, they are (u,v) from instance.
                # instance returns sorted edges.
                # If u < v, edge is (u, v). If u > v, edge is (v, u)?
                # In instance: 
                # C-C: u < v.
                # C-I: u in 0..k-1, v in k..n-1. So u < v always.
                # So edges are (u, v) with u < v.
                edge = (u, v)
                if edge in all_edges and edge not in covered:
                    U_v.add(u)
            
            if not U_v:
                continue
            
            clique = [v] + list(U_v)
            cliques.append(clique)
            
            # Mark edges in this clique as covered
            # Clique is {v} U U_v.
            # Edges: (u, v) for u in U_v.
            # Edges: (u1, u2) for u1, u2 in U_v, u1 < u2.
            for u in U_v:
                covered.add((u, v))
            U_v_list = list(U_v)
            for i in range(len(U_v_list)):
                for j in range(i+1, len(U_v_list)):
                    a, b = U_v_list[i], U_v_list[j]
                    edge = (min(a, b), max(a, b))
                    if edge in all_edges: # Should be, since C is a clique
                        covered.add(edge)
                        
        # Step 2: Cover remaining C edges
        remaining_edges = C_edges - covered
        
        if not remaining_edges:
            return cliques
        
        # Build adjacency for remaining graph on C
        adj = [set() for _ in range(k)]
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Color complement graph
        comp_adj = [set() for _ in range(k)]
        for u in range(k):
            for v in range(u+1, k):
                if (u, v) not in remaining_edges:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
                    
        colors = [-1] * k
        color_classes = []
        
        for u in range(k):
            used_colors = set()
            for v in comp_adj[u]:
                if colors[v] != -1:
                    used_colors.add(colors[v])
            c = 0
            while c in used_colors:
                c += 1
            colors[u] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(u)
            
        for cc in color_classes:
            if len(cc) >= 2:
                cliques.append(list(cc))
                
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        cliques = solve_split(*spec)
        # Ensure types are correct
        results.append([list(clique) for clique in cliques])
    return results
