# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, d, seed):
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

    def build_conflict_graph(edges, adj):
        n = len(edges)
        conflict_adj = [set() for _ in range(n)]
        for i in range(n):
            u1, v1 = edges[i]
            for j in range(i + 1, n):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
        return conflict_adj

    def dsatur_coloring(conflict_adj, num_vertices, rng):
        colors = [-1] * num_vertices
        sat = [set() for _ in range(num_vertices)]
        degrees = [len(conflict_adj[i]) for i in range(num_vertices)]
        uncolored = list(range(num_vertices))
        
        while uncolored:
            best_v = -1
            best_sat = -1
            best_deg = -1
            # Add slight randomness to tie-breaking to allow restarts
            for v in uncolored:
                s = len(sat[v])
                d = degrees[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v
            
            used_colors = sat[best_v]
            c = 0
            while c in used_colors:
                c += 1
            
            colors[best_v] = c
            uncolored.remove(best_v)
            
            for u in conflict_adj[best_v]:
                if u in uncolored:
                    sat[u].add(c)

        return colors

    def local_search_improve(colors, conflict_adj, num_vertices, rng):
        """
        Improves coloring by trying to reduce the number of colors.
        """
        if num_vertices == 0:
            return colors
        
        max_color = max(colors)
        # We try to eliminate colors from max_color down to 0
        # We iterate multiple times to allow cascading improvements
        improved = True
        iterations = 0
        max_iterations = 50 # Safety limit
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            # Re-calculate max_color as it might decrease
            if not any(c > max_color for c in colors):
                # Check if we can go lower
                for c in range(max_color, -1, -1):
                    if c not in colors:
                        max_color = c - 1
                        break
                if max_color < 0:
                    break
            
            # Try to recolor vertices of the current max_color to lower colors
            # We collect vertices of max_color
            vertices_of_max = [v for v in range(num_vertices) if colors[v] == max_color]
            
            for v in vertices_of_max:
                # Find a valid lower color
                # Forbidden colors are those used by neighbors
                forbidden = set()
                for u in conflict_adj[v]:
                    if colors[u] < max_color:
                        forbidden.add(colors[u])
                
                # Try colors 0 to max_color - 1
                new_c = -1
                for c in range(max_color):
                    if c not in forbidden:
                        new_c = c
                        break
                
                if new_c != -1:
                    colors[v] = new_c
                    improved = True
                    # After recoloring, the max_color might drop if no other vertex has max_color
                    # We can break and restart the loop to recalculate max_color
                    break # Restart the outer while loop to recalculate state
        
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n = len(edges)
        
        # Build original graph adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build conflict graph
        conflict_adj = build_conflict_graph(edges, adj)
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Run multiple strategies/restarts
        # Strategy 1: DSATUR with fixed seed
        rng1 = Random(42)
        colors1 = dsatur_coloring(conflict_adj, n, rng1)
        colors1 = local_search_improve(colors1, conflict_adj, n, rng1)
        nc1 = max(colors1) + 1 if n > 0 else 0
        
        if nc1 < best_num_colors:
            best_num_colors = nc1
            best_colors = colors1
            
        # Strategy 2: DSATUR with random restarts
        for i in range(10):
            rng_r = Random(100 + i)
            # Perturb the graph or the process? 
            # Since DSATUR is deterministic given the graph, we need to randomize the tie-breaking or the order.
            # The current dsatur_coloring uses a deterministic loop over uncolored. 
            # To make it random, we can shuffle the initial uncolored list or the neighbor iteration.
            # Let's just use a simple greedy with random order as a second strategy.
            
            # Greedy with random order
            order = list(range(n))
            rng_r.shuffle(order)
            # Sort by degree descending, but with random ties
            degrees = [len(conflict_adj[i]) for i in range(n)]
            # Add random noise to degree for sorting
            noisy_degrees = [degrees[i] + rng_r.random() * 0.1 for i in range(n)]
            order = sorted(range(n), key=lambda x: noisy_degrees[x], reverse=True)
            
            colors_g = [-1] * n
            forbidden = [set() for _ in range(n)]
            for v in order:
                c = 0
                while c in forbidden[v]:
                    c += 1
                colors_g[v] = c
                for u in conflict_adj[v]:
                    forbidden[u].add(c)
            
            colors_g = local_search_improve(colors_g, conflict_adj, n, rng_r)
            nc_g = max(colors_g) + 1 if n > 0 else 0
            
            if nc_g < best_num_colors:
                best_num_colors = nc_g
                best_colors = colors_g

        out.append(best_colors)

    return out
# EVOLVE-BLOCK-END
