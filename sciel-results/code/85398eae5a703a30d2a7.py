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
        num_edges = len(edges)
        if num_edges == 0:
            return []

        # Build adjacency list for vertices
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict masks
        # Edge i conflicts with Edge j if they share a vertex or are distance 2 apart
        conflict_mask = [0] * num_edges
        
        for i in range(num_edges):
            u, v = edges[i]
            blocking_verts = {u, v}
            blocking_verts.update(adj[u])
            blocking_verts.update(adj[v])
            
            for j in range(num_edges):
                if i == j:
                    continue
                x, y = edges[j]
                if x in blocking_verts or y in blocking_verts:
                    conflict_mask[i] |= (1 << j)

        color_of_edge = [-1] * num_edges
        uncolored_mask = (1 << num_edges) - 1
        
        current_color = 0
        
        while uncolored_mask:
            # Start a new color class
            available_mask = uncolored_mask
            current_class_edges = []
            
            while available_mask:
                # Find the edge in available_mask with the highest number of conflicts
                # with other edges in available_mask.
                # This is a greedy heuristic to "cover" more edges.
                
                best_edge = -1
                max_conflicts = -1
                
                # Iterate through set bits in available_mask
                # We can do this efficiently by checking bits
                temp_mask = available_mask
                while temp_mask:
                    # Get lowest set bit
                    lsb = temp_mask & -temp_mask
                    idx = lsb.bit_length() - 1
                    
                    # Count conflicts with other available edges
                    # conflicts = conflict_mask[idx] & available_mask
                    # We exclude idx itself, but it's not in conflict_mask[idx] usually?
                    # Yes, conflict_mask[i] does not have bit i set.
                    conflicts = conflict_mask[idx] & available_mask
                    count = bin(conflicts).count('1')
                    
                    if count > max_conflicts:
                        max_conflicts = count
                        best_edge = idx
                    
                    temp_mask ^= lsb
                
                # Add best_edge to current class
                current_class_edges.append(best_edge)
                
                # Remove best_edge and all its conflicting neighbors from available_mask
                # The neighbors to remove are best_edge itself and all edges in conflict_mask[best_edge]
                remove_mask = (1 << best_edge) | (conflict_mask[best_edge] & available_mask)
                available_mask &= ~remove_mask
                
            # Assign color to all edges in current_class_edges
            for idx in current_class_edges:
                color_of_edge[idx] = current_color
                uncolored_mask &= ~(1 << idx)
                
            current_color += 1
            
        return color_of_edge

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = strong_edge_coloring(edges, n)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
