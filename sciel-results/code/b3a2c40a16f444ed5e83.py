import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_triangle_free_bitmask(adj_mask):
        """Check triangle-free using bitmasks: no vertex u,v,w all mutually connected."""
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    if adj_mask[u] & adj_mask[v]:
                        return False
        return True
    
    def is_3_colorable_bitmask(adj_mask):
        """Check 3-colorability using backtracking with bitmask color tracking."""
        # Precompute neighbor lists from bitmask
        neighbors = []
        for u in range(n):
            nbrs = []
            for v in range(n):
                if adj_mask[u] & (1 << v):
                    nbrs.append(v)
            neighbors.append(nbrs)
        
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: -len(neighbors[x]))
        
        # color_mask[v] = bitmask of colors used by already-colored neighbors of v
        # color[v] = assigned color (-1 if uncolored)
        color = [-1] * n
        color_mask = [0] * n  # bitmask of forbidden colors for each vertex
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            forbidden = color_mask[v]
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    # Update color_mask for uncolored neighbors
                    updated = []
                    for u in neighbors[v]:
                        if color[u] == -1:
                            color_mask[u] |= (1 << c)
                            updated.append(u)
                    if backtrack(idx + 1):
                        return True
                    # Undo
                    color[v] = -1
                    for u in updated:
                        color_mask[u] &= ~(1 << c)
            return False
        
        return backtrack(0)
    
    def edge_count(adj_mask):
        count = 0
        for i in range(n):
            count += bin(adj_mask[i]).count('1')
        return count // 2
    
    def groetzsch_graph():
        """Grötzsch graph: Mycielski of C5, 11 vertices, 20 edges, 4-chromatic triangle-free."""
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        edges += [(5,1),(5,4),(6,2),(6,0),(7,3),(7,1),(8,4),(8,2),(9,0),(9,3)]
        edges += [(10,5),(10,6),(10,7),(10,8),(10,9)]
        return edges
    
    def build_adj_mask(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def greedy_add_edges(adj_mask, deadline):
        """Greedily add edges maintaining triangle-free and non-3-colorable.
        Prioritize connecting low-degree vertices to independent sets."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute degrees
            degrees = [bin(adj_mask[u]).count('1') for u in range(n)]
            
            # Build priority list: sort non-edges by (min_degree, max_degree) ascending
            # to connect isolated/low-degree vertices first
            non_edges = []
            for u, v in all_pairs:
                if not (adj_mask[u] & (1 << v)):
                    # Check triangle-free
                    if adj_mask[u] & adj_mask[v]:
                        continue
                    non_edges.append((min(degrees[u], degrees[v]), max(degrees[u], degrees[v]), u, v))
            
            # Sort by priority: lower degree vertices first
            non_edges.sort(key=lambda x: (x[0], x[1]))
            
            for _, _, u, v in non_edges:
                if time.time() >= deadline:
                    break
                # Try adding edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                # Check 3-colorability
                if not is_3_colorable_bitmask(adj_mask):
                    improved = True
                    # Update degrees
                    degrees[u] += 1
                    degrees[v] += 1
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
        return adj_mask
    
    def local_search_swap(adj_mask, deadline):
        """Try 1-1 swaps: remove one edge, add another, maintaining constraints.
        Focus on removing edges from high-degree vertices to create space for better additions."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj_mask[u] & (1 << v):
                        current_edges.append((u, v))
            
            # Sort edges by sum of degrees descending - try removing from dense areas first
            degrees = [bin(adj_mask[u]).count('1') for u in range(n)]
            current_edges.sort(key=lambda e: -(degrees[e[0]] + degrees[e[1]]))
            
            for u, v in current_edges:
                if time.time() >= deadline:
                    return adj_mask, False
                # Remove edge (u,v)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                found_swap = False
                # Try adding edges that connect to the freed-up vertices
                candidate_adds = []
                for a, b in all_pairs:
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    if adj_mask[a] & (1 << b):
                        continue
                    # Check triangle-free
                    if adj_mask[a] & adj_mask[b]:
                        continue
                    # Prioritize edges that connect to u or v
                    score = 0
                    if a == u or b == u or a == v or b == v:
                        score = 1
                    candidate_adds.append((score, a, b))
                
                # Sort: prioritize edges connecting to freed vertices
                candidate_adds.sort(key=lambda x: -x[0])
                
                for _, a, b in candidate_adds:
                    if time.time() >= deadline:
                        break
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    if not is_3_colorable_bitmask(adj_mask):
                        improved = True
                        found_swap = True
                        break
                    adj_mask[a] &= ~(1 << b)
                    adj_mask[b] &= ~(1 << a)
                
                if found_swap:
                    return adj_mask, True
                
                # Restore edge (u,v)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        return adj_mask, False
    
    # Start with Grötzsch graph on vertices 0-10, vertices 11-15 isolated
    initial_edges = groetzsch_graph()
    best_adj = build_adj_mask(initial_edges)
    best_count = edge_count(best_adj)
    
    # First, greedily add edges with priority on low-degree vertices
    best_adj = greedy_add_edges(best_adj, deadline)
    best_count = edge_count(best_adj)
    
    # Then do local search with multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        adj = best_adj[:]
        
        # Do local search
        adj, improved = local_search_swap(adj, deadline)
        
        if improved:
            new_count = edge_count(adj)
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
                # After improvement, do another greedy pass to fill in
                if time.time() < deadline:
                    best_adj = greedy_add_edges(best_adj, deadline)
                    best_count = edge_count(best_adj)
    
    # Final greedy pass
    best_adj = greedy_add_edges(best_adj, deadline)
    best_count = edge_count(best_adj)
    
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                final_edges.append([i, j])
    
    return final_edges
