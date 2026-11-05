"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring(edges, n):
        """
        Colors edges such that each color class is an induced matching.
        Two edges conflict if they share a vertex or are joined by an edge.
        Uses DSATUR heuristic with bitset conflict checking.
        """
        num_edges = len(edges)
        if num_edges == 0:
            return []

        # Build adjacency list for vertices
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict bitmasks for each edge.
        # Edge i conflicts with Edge j if:
        # 1. They share a vertex.
        # 2. They are connected by an edge (distance 2 in the line graph sense).
        
        conflict_mask = [0] * num_edges
        
        for i in range(num_edges):
            u, v = edges[i]
            # Create a set of vertices that "block" this edge.
            # If another edge has an endpoint in this set, it conflicts.
            # The set is {u, v} U N(u) U N(v).
            blocking_verts = {u, v}
            blocking_verts.update(adj[u])
            blocking_verts.update(adj[v])
            
            # Now, for every other edge j, if it touches any vertex in blocking_verts, it conflicts.
            for j in range(num_edges):
                if i == j:
                    continue
                x, y = edges[j]
                if x in blocking_verts or y in blocking_verts:
                    conflict_mask[i] |= (1 << j)

        # DSATUR Heuristic:
        # We select the uncolored edge with the highest saturation (number of distinct colors in its neighborhood),
        # breaking ties by the highest degree (number of conflicts).
        
        color_of_edge = [-1] * num_edges
        color_masks = [0] # color_masks[c] is bitmask of edges with color c
        
        # Precompute degree (number of conflicts)
        degrees = [bin(mask).count('1') for mask in conflict_mask]
        
        uncolored = set(range(num_edges))
        
        while uncolored:
            # Find the edge with max saturation, then max degree
            best_edge = None
            best_sat = -1
            best_deg = -1
            
            for i in uncolored:
                # Calculate saturation: number of distinct colors in neighbors
                sat = 0
                for c in range(len(color_masks)):
                    if conflict_mask[i] & color_masks[c]:
                        sat += 1
                
                deg = degrees[i]
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            # Assign best_edge a color
            # Find smallest color c such that no conflict
            assigned_color = -1
            for c in range(len(color_masks)):
                if not (conflict_mask[best_edge] & color_masks[c]):
                    assigned_color = c
                    break
            
            if assigned_color == -1:
                # New color
                assigned_color = len(color_masks)
                color_masks.append(0)
            
            color_of_edge[best_edge] = assigned_color
            color_masks[assigned_color] |= (1 << best_edge)
            uncolored.remove(best_edge)
            
        return color_of_edge

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = strong_edge_coloring(edges, n)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
