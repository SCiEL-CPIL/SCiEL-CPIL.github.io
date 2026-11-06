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
        """Fast 3-colorability check using MRV backtracking."""
        colors = [-1] * n
        
        def backtrack():
            # Find unassigned vertex with min remaining values (MRV)
            best_v = -1
            min_avail = 4
            forbidden_best = 0
            
            for v in range(n):
                if colors[v] != -1:
                    continue
                forbidden = 0
                m = adj_mask[v]
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m -= lsb
                    cu = colors[u]
                    if cu != -1:
                        forbidden |= (1 << cu)
                avail = 3 - bin(forbidden).count('1')
                if avail < min_avail:
                    min_avail = avail
                    best_v = v
                    forbidden_best = forbidden
                    if min_avail == 0:
                        break
            
            if best_v == -1:
                return True # All colored
            
            if min_avail == 0:
                return False
            
            for c in range(3):
                if not (forbidden_best & (1 << c)):
                    colors[best_v] = c
                    if backtrack():
                        return True
                    colors[best_v] = -1
            return False
        
        return backtrack()
    
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
        """Greedily add edges maintaining triangle-free. 
        Assumes input is already 4-chromatic, so 3-colorability is preserved."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all non-edges in random order
            non_edges = []
            for u, v in all_pairs:
                if not (adj_mask[u] & (1 << v)):
                    non_edges.append((u, v))
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() >= deadline:
                    break
                # Check triangle-free: no common neighbor
                if adj_mask[u] & adj_mask[v]:
                    continue
                # Try adding edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                # No 3-color check needed as adding edges preserves 4-chromaticity
                improved = True
        return adj_mask
    
    def local_search_swap(adj_mask, deadline):
        """Try 1-1 swaps: remove one edge, add another, maintaining constraints."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if adj_mask[u] & (1 << v):
                        current_edges.append((u, v))
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() >= deadline:
                    return adj_mask, False
                # Remove edge (u,v)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                # Check if graph is still 4-chromatic after removal
                still_4_chromatic = not is_3_colorable_bitmask(adj_mask)
                
                found_swap = False
                for a, b in all_pairs:
                    if time.time() >= deadline:
                        break
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    if adj_mask[a] & (1 << b):
                        continue
                    # Check triangle-free
                    if adj_mask[a] & adj_mask[b]:
                        continue
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    
                    valid = False
                    if still_4_chromatic:
                        valid = True
                    else:
                        valid = not is_3_colorable_bitmask(adj_mask)
                        
                    if valid:
                        improved = True
                        found_swap = True
                        break
                    else:
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
    
    # First, greedily add edges
    best_adj = greedy_add_edges(best_adj, deadline)
    best_count = edge_count(best_adj)
    
    # Then do local search with multiple restarts
    max_restarts = 30
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
