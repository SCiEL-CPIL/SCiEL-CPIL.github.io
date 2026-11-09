import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_edges_set = set(all_edges)
    
    def edge_to_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
    
    def idx_to_edge(idx):
        # Inverse of edge_to_idx
        for u in range(n):
            for v in range(u+1, n):
                if edge_to_idx(u, v) == idx:
                    return (u, v)
        return None
    
    # Precompute: for each edge, which other edges are compatible (no triangle, no shared vertex)
    # Two edges (a,b) and (c,d) are compatible if they don't share a vertex and don't form a triangle
    # Actually, for triangle-free: edges (a,b) and (c,d) are incompatible if:
    #   - they share a vertex (a==c, a==d, b==c, b==d) -- but this is just "can't both be in matching"
    #   Wait, no. In a graph, two edges can share a vertex. The triangle constraint is:
    #   if we have edges (a,b), (b,c), (a,c), that's a triangle.
    #   So edge (a,b) is incompatible with edge (c,d) only if {a,b,c,d} has a triangle formed.
    #   Actually, adding edge (a,b) creates a triangle if there exists c such that (a,c) and (b,c) are both present.
    #   So the conflict is: edge (a,b) conflicts with the pair {(a,c), (b,c)} for any c.
    #   But for a pairwise conflict graph where nodes are edges: two edges e1, e2 are "conflicting" if
    #   they cannot both be in a triangle-free graph. Two edges can always both be in a triangle-free graph
    #   unless... actually any two edges can coexist in a triangle-free graph (just don't add the third edge).
    #   
    #   So the real constraint is: a SET of edges is triangle-free iff for every triple of vertices,
    #   not all three edges are present. This is a 3-way constraint, not pairwise.
    #   
    #   So the maximum clique approach on a pairwise conflict graph doesn't directly work for triangle-free.
    #   Let me reconsider.
    #
    #   Alternative approach: We want to maximize edges in a triangle-free, non-3-colorable graph.
    #   Key insight: A graph is triangle-free iff its edges form a graph with no K3.
    #   We can use the following: for a triangle-free graph, by Turan's theorem, the max edges is floor(n^2/4) = 64 for n=16.
    #   But we also need it to NOT be 3-colorable.
    #
    #   Let me use a different strategy: 
    #   1. Start with a known 4-chromatic triangle-free graph on 16 vertices
    #   2. Use local search to maximize edges while maintaining triangle-free and non-3-colorable
    
    # Better 3-colorability check using bitmask with forward checking
    def is_3_colorable(adj_mask):
        """Fast 3-colorability check with MRV + forward checking."""
        color = [-1] * n
        
        def backtrack():
            # Find uncolored vertex with fewest available colors (MRV with forward checking)
            best_v = -1
            best_avail = 4
            for v in range(n):
                if color[v] == -1:
                    used = 0
                    m = adj_mask[v]
                    while m:
                        lsb = m & (-m)
                        u = lsb.bit_length() - 1
                        if color[u] >= 0:
                            used |= (1 << color[u])
                        m ^= lsb
                    avail = 3 - bin(used & 0b111).count('1')
                    if avail < best_avail:
                        best_avail = avail
                        best_v = v
                        if avail == 0:
                            return False  # Forward checking: dead end
                        if avail == 1:
                            break
            
            if best_v == -1:
                return True
            
            v = best_v
            used = 0
            m = adj_mask[v]
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                if color[u] >= 0:
                    used |= (1 << color[u])
                m ^= lsb
            
            # Try colors in order of most constrained first
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = -1
            return False
        
        return backtrack()
    
    def edge_count(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def is_triangle_free(adj_mask):
        """Check if graph is triangle-free using bitmasks."""
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                    if common:
                        return False
        return True
    
    def is_valid(adj_mask):
        """Check triangle-free and not 3-colorable."""
        if not is_triangle_free(adj_mask):
            return False
        return not is_3_colorable(adj_mask)
    
    # Build initial graph: Mycielski of C5 (Grötzsch) extended to 16 vertices
    # Start with C5 on vertices 0-4
    adj_mask = [0] * n
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(adj_mask, i, (i+1) % 5)
    
    # Mycielski construction: add vertices 5-9 as copies of 0-4, vertex 10 as apex
    # For each edge (u,v) in C5, add edges (u, 5+v) and (v, 5+u)
    for i in range(5):
        u, v = i, (i+1) % 5
        add_edge(adj_mask, u, 5+v)
        add_edge(adj_mask, v, 5+u)
    
    # Connect 5-i to 10 for i=0..4
    for i in range(5):
        add_edge(adj_mask, 5+i, 10)
    
    # Now we have 11 vertices. Add vertices 11-15 using MIS-based strategy.
    # For each new vertex, connect it to a large independent set in the current graph
    # to maximize edges while maintaining triangle-free property.
    
    def find_max_independent_set(adj_mask, exclude=None):
        """Find a large independent set in the current graph (greedy + local improvement)."""
        if exclude is None:
            exclude = set()
        
        # Greedy: pick vertices with low degree first
        remaining = [v for v in range(n) if v not in exclude]
        random.shuffle(remaining)
        
        # Sort by degree (ascending) for greedy
        remaining.sort(key=lambda v: bin(adj_mask[v] & ~(1 << v)).count('1'))
        
        mis = []
        used = 0
        for v in remaining:
            if not (adj_mask[v] & used):
                mis.append(v)
                used |= (1 << v)
        
        # Local improvement: try to add more vertices
        improved = True
        while improved:
            improved = False
            current_mask = 0
            for v in mis:
                current_mask |= (1 << v)
            
            for v in range(n):
                if v in exclude or (current_mask & (1 << v)):
                    continue
                if not (adj_mask[v] & current_mask):
                    mis.append(v)
                    current_mask |= (1 << v)
                    improved = True
                    break
        
        return mis
    
    # Add vertices 11-15 one by one, connecting each to a large MIS
    for new_v in range(11, 16):
        # Find MIS in the current graph (excluding new_v since it's not connected yet)
        exclude_set = set()
        mis = find_max_independent_set(adj_mask, exclude_set)
        
        # Connect new_v to all vertices in the MIS
        for v in mis:
            if v != new_v and not (adj_mask[new_v] & (1 << v)):
                # Check triangle-free: new_v and v must have no common neighbor
                common = adj_mask[new_v] & adj_mask[v] & ~(1 << new_v) & ~(1 << v)
                if not common:
                    add_edge(adj_mask, new_v, v)
    
    best_adj = list(adj_mask)
    best_count = edge_count(best_adj)
    
    # Phase 1: Greedy edge addition
    def greedy_add(adj_mask, max_time):
        """Try to add edges greedily."""
        changed = True
        while changed and time.time() < max_time:
            changed = False
            # Shuffle edge order for diversity
            edges = list(all_edges)
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > max_time:
                    break
                if adj_mask[u] & (1 << v):
                    continue
                # Check triangle-free: no common neighbor
                common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                # Try adding
                new_adj = list(adj_mask)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                # Valid
                add_edge(adj_mask, u, v)
                changed = True
        return adj_mask
    
    adj_mask = greedy_add(adj_mask, deadline - 40)
    current_count = edge_count(adj_mask)
    if current_count > best_count:
        best_count = current_count
        best_adj = list(adj_mask)
    
    # Phase 2: Local search with edge swaps (more aggressive)
    def local_search(adj_mask, max_time):
        """Local search: try edge swaps (remove one, add another)."""
        current = list(adj_mask)
        current_count = edge_count(current)
        best_local = list(current)
        best_local_count = current_count
        
        no_improve = 0
        max_no_improve = 5000  # Increased from 200
        
        while time.time() < max_time and no_improve < max_no_improve:
            # Get existing edges
            existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if current[i] & (1 << j)]
            if not existing:
                break
            
            # Pick random edge to remove
            ui, vi = random.choice(existing)
            remove_edge(current, ui, vi)
            
            # Try to add a different edge
            added = False
            candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                         if not (current[i] & (1 << j)) and (i,j) != (ui, vi)]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if time.time() > max_time:
                    break
                common = current[u] & current[v] & ~(1 << u) & ~(1 << v)
                if common:
                    continue
                new_adj = list(current)
                add_edge(new_adj, u, v)
                if is_3_colorable(new_adj):
                    continue
                add_edge(current, u, v)
                added = True
                break
            
            if not added:
                # Put the edge back
                add_edge(current, ui, vi)
                no_improve += 1
                continue
            
            new_count = edge_count(current)
            if new_count > best_local_count:
                best_local_count = new_count
                best_local = list(current)
                no_improve = 0
            else:
                no_improve += 1
        
        return best_local, best_local_count
    
    adj_mask, local_count = local_search(adj_mask, deadline - 20)
    if local_count > best_count:
        best_count = local_count
        best_adj = list(adj_mask)
    
    # Phase 3: Multi-start with perturbation
    for restart in range(10):
        if time.time() > deadline - 15:
            break
        
        # Start from best, perturb by removing some edges
        current = list(best_adj)
        existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if current[i] & (1 << j)]
        random.shuffle(existing)
        num_remove = min(5, len(existing))
        for edge in existing[:num_remove]:
            remove_edge(current, edge[0], edge[1])
        
        # Greedy add
        current = greedy_add(current, deadline - 15 - restart * 1)
        current_count = edge_count(current)
        
        # Local search
        current, ls_count = local_search(current, deadline - 10 - restart * 1)
        if ls_count > best_count:
            best_count = ls_count
            best_adj = list(current)
    
    # Final greedy cleanup
    final_adj = list(best_adj)
    edges = list(all_edges)
    random.shuffle(edges)
    for u, v in edges:
        if time.time() > deadline - 1:
            break
        if final_adj[u] & (1 << v):
            continue
        common = final_adj[u] & final_adj[v] & ~(1 << u) & ~(1 << v)
        if common:
            continue
        new_adj = list(final_adj)
        add_edge(new_adj, u, v)
        if is_3_colorable(new_adj):
            continue
        add_edge(final_adj, u, v)
    
    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj
    
    # Return result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
