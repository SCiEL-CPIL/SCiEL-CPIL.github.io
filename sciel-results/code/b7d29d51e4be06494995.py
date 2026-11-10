import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def get_initial_upper_bound(adj, n):
        # Greedy vertex cover
        cover = set()
        remaining_adj = [adj[i] for i in range(n)]
        remaining_vertices = [i for i in range(n) if remaining_adj[i] != 0]
        
        while remaining_vertices:
            # Pick vertex with max degree in remaining graph
            max_deg = -1
            max_v = -1
            for v in remaining_vertices:
                deg = bin(remaining_adj[v]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    max_v = v
            
            if max_v == -1:
                break
                
            cover.add(max_v)
            
            # Remove max_v and its neighbors from consideration
            # Update remaining_adj
            neighbors = bin(remaining_adj[max_v]).count('1')
            # We need to remove edges incident to max_v
            # And remove max_v from the list of active vertices
            
            # Identify neighbors to remove from remaining_vertices
            # Actually, we just need to clear edges.
            
            # Let's just rebuild the active set logic carefully.
            # For simplicity in the upper bound heuristic, we can just iterate.
            
            # Remove max_v from remaining_vertices
            remaining_vertices.remove(max_v)
            
            # For all u in remaining_vertices, clear edge to max_v
            for u in remaining_vertices:
                if (remaining_adj[u] >> max_v) & 1:
                    remaining_adj[u] &= ~(1 << max_v)
                    remaining_adj[max_v] &= ~(1 << u) # Not strictly needed if max_v is removed, but good for consistency
            
            # Remove vertices that became isolated
            new_remaining = []
            for u in remaining_vertices:
                if remaining_adj[u] != 0:
                    new_remaining.append(u)
            remaining_vertices = new_remaining
            
        return len(cover), cover

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        
        # Filter to active vertices (degree >= 1)
        active_vertices = [i for i in range(n) if adj[i] != 0]
        if not active_vertices:
            return []
            
        # Map original indices to 0..m-1 for compact representation
        m = len(active_vertices)
        index_map = {v: i for i, v in enumerate(active_vertices)}
        
        # Build compact adjacency
        c_adj = [0] * m
        for u in active_vertices:
            i = index_map[u]
            for v in active_vertices:
                if v != u and (adj[u] >> v) & 1:
                    j = index_map[v]
                    c_adj[i] |= (1 << j)
                    
        # Initial upper bound
        ub_size, ub_cover = get_initial_upper_bound(c_adj, m)
        best_size = ub_size
        best_cover = set(ub_cover)
        
        # Branch and Bound
        # State: active_mask (bitmask of vertices still in graph), current_cover (list of compact indices)
        # We represent the graph by the active_mask and the original c_adj.
        # An edge (i, j) is active if i and j are in active_mask and c_adj[i] has bit j.
        
        def branch_and_bound(active_mask, current_cover, depth):
            nonlocal best_size, best_cover
            
            if time.time() > deadline:
                return
            
            # Check for termination
            # If no edges in active_mask, we have a complete cover
            has_edges = False
            for i in range(m):
                if (active_mask >> i) & 1:
                    # Check if i has any active neighbors
                    if c_adj[i] & active_mask:
                        has_edges = True
                        break
            
            if not has_edges:
                if len(current_cover) < best_size:
                    best_size = len(current_cover)
                    best_cover = set(current_cover)
                return
            
            # Lower Bound: Size of a maximal matching
            # Greedy matching on active subgraph
            matched = [False] * m
            match_size = 0
            for i in range(m):
                if not matched[i] and (active_mask >> i) & 1:
                    # Find a neighbor j
                    neighbors = c_adj[i] & active_mask
                    # We need to pick a neighbor not yet matched
                    # This simple greedy might not find max matching, but it's a lower bound
                    # To make it a valid lower bound, we just need a matching.
                    # A maximal matching is a matching.
                    for j in range(m):
                        if (neighbors >> j) & 1 and not matched[j]:
                            matched[i] = True
                            matched[j] = True
                            match_size += 1
                            break
            
            if len(current_cover) + match_size >= best_size:
                return
            
            # Branching: Pick an edge
            # Find an active edge
            edge_u = -1
            edge_v = -1
            max_deg = -1
            for i in range(m):
                if (active_mask >> i) & 1:
                    deg = bin(c_adj[i] & active_mask).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        # Pick a neighbor
                        neighbors = c_adj[i] & active_mask
                        # Pick first neighbor
                        lsb = neighbors & -neighbors
                        j = lsb.bit_length() - 1
                        edge_u = i
                        edge_v = j
                        break
            
            if edge_u == -1:
                return # No edges, handled above
            
            # Branch 1: Include edge_u
            new_active_mask_1 = active_mask & ~(1 << edge_u)
            # Remove edges incident to edge_u
            # We don't modify c_adj, we just mask out edge_u from active_mask
            # But we also need to ensure that edges *from* edge_u are not considered.
            # Since we check `c_adj[i] & active_mask`, if edge_u is not in active_mask, 
            # it won't be considered as a source. 
            # However, if i is in active_mask, and c_adj[i] has edge_u, but edge_u is not in active_mask,
            # the bit is effectively ignored.
            
            current_cover.append(edge_u)
            branch_and_bound(new_active_mask_1, current_cover, depth + 1)
            current_cover.pop()
            
            # Branch 2: Include edge_v
            new_active_mask_2 = active_mask & ~(1 << edge_v)
            current_cover.append(edge_v)
            branch_and_bound(new_active_mask_2, current_cover, depth + 1)
            current_cover.pop()
            
        initial_active_mask = (1 << m) - 1
        branch_and_bound(initial_active_mask, [], 0)
        
        # Map back to original indices
        final_cover = [active_vertices[i] for i in best_cover]
        return sorted(final_cover)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback to simple heuristic if time runs out
            adj = random_graph(n, p, seed)
            touched = {x for e in [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1] for x in e}
            out.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        out.append(res)
        
    return out
