"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    def instance(n, d, seed):
        from random import Random
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def strong_color(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph (L^2(G))
        adj_set = set()
        for u, nbrs in adj.items():
            for v in nbrs:
                adj_set.add((u, v))
                adj_set.add((v, u))
        
        # edge_adj[i] is a set of indices j such that edges[i] and edges[j] conflict
        edge_adj = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    edge_adj[i].add(j)
                    edge_adj[j].add(i)
                    continue
                if (u1, u2) in adj_set or (u1, v2) in adj_set or (v1, u2) in adj_set or (v1, v2) in adj_set:
                    edge_adj[i].add(j)
                    edge_adj[j].add(i)
        
        # Degrees in conflict graph
        deg = [len(nei) for nei in edge_adj]
        
        # Greedy DSATUR to get an initial upper bound
        def greedy_dsatur():
            color = [-1] * m
            neighbor_colors = [set() for _ in range(m)]
            uncolored = list(range(m))
            num_colors = 0
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(neighbor_colors[v])
                    d_v = deg[v]
                    if sat > best_sat or (sat == best_sat and d_v > best_deg):
                        best_sat = sat
                        best_deg = d_v
                        best_v = v
                used = neighbor_colors[best_v]
                c = 0
                while c in used:
                    c += 1
                color[best_v] = c
                if c + 1 > num_colors:
                    num_colors = c + 1
                for nb in edge_adj[best_v]:
                    if nb in uncolored:
                        neighbor_colors[nb].add(c)
                uncolored.remove(best_v)
            return color, num_colors

        init_color, ub = greedy_dsatur()
        
        # Backtracking search to find minimum colors
        # We search for a coloring with K colors, starting from lb to ub
        # Lower bound: max(deg) + 1
        lb = max(deg) + 1
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(K):
            # Returns a valid coloring list if possible with K colors, else None
            color = [-1] * m
            neighbor_colors = [set() for _ in range(m)]
            uncolored = list(range(m))
            
            # Order vertices statically by degree to help pruning? 
            # Dynamic DSATUR is better.
            
            def recursive_solve():
                if not uncolored:
                    return True
                
                # Select vertex with max saturation, then max degree
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(neighbor_colors[v])
                    d_v = deg[v]
                    if sat > best_sat or (sat == best_sat and d_v > best_deg):
                        best_sat = sat
                        best_deg = d_v
                        best_v = v
                
                # Forward checking: Check if any uncolored neighbor has 0 available colors
                # Actually, we just try colors for best_v.
                # To prune, we can check if any other uncolored vertex has no available colors.
                # This is expensive. Let's stick to standard backtracking with good ordering.
                
                for c in range(K):
                    if c not in neighbor_colors[best_v]:
                        color[best_v] = c
                        # Update neighbors
                        for nb in edge_adj[best_v]:
                            if nb in uncolored:
                                neighbor_colors[nb].add(c)
                        
                        # Check for immediate failure: any uncolored neighbor with no options?
                        valid = True
                        for nb in edge_adj[best_v]:
                            if nb in uncolored:
                                if len(neighbor_colors[nb]) == K:
                                    valid = False
                                    break
                            # Also check if nb has no available colors at all?
                            # neighbor_colors[nb] contains colors used by *colored* neighbors.
                            # If len(neighbor_colors[nb]) == K, then no color is available.
                            # Wait, neighbor_colors only tracks colors used by *colored* neighbors.
                            # If a neighbor is uncolored, its color is not in neighbor_colors.
                            # So if len(neighbor_colors[nb]) == K, it means all K colors are used by its colored neighbors.
                            # Thus, it has no valid color.
                        
                        if valid:
                            uncolored.remove(best_v)
                            if recursive_solve():
                                return True
                            uncolored.append(best_v)
                        
                        # Undo
                        for nb in edge_adj[best_v]:
                            if nb in uncolored:
                                neighbor_colors[nb].remove(c)
                        color[best_v] = -1
                        
                return False
            
            if recursive_solve():
                return color
            return None

        best_color = None
        # Try to find a coloring with fewer colors than the greedy upper bound
        # Start from lb, go up to ub
        # If lb == ub, we are done.
        # Usually lb is close to ub.
        for K in range(lb, ub + 1):
            res = backtrack(K)
            if res is not None:
                best_color = res
                break
        
        if best_color is None:
            best_color = init_color
        
        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        colors = strong_color(edges, adj)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
