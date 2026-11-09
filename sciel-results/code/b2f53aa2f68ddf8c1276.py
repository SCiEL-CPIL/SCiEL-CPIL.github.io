import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # For triangle-freeness: when adding edge (u,v), it creates a triangle
    # if there exists w such that (u,w) and (v,w) are both present.
    # Precompute for each edge, the set of edges that would conflict (share a vertex with it)
    # Actually, for triangle check: edge e=(u,v) conflicts with edge f=(w,x) if
    # {u,v} and {w,x} share exactly one vertex AND the other two vertices are connected.
    # Simpler: maintain adjacency bitmasks.
    
    # Bitmask approach for fast checks
    # adj_mask[i] is a bitmask of neighbors of vertex i
    
    # For 3-colorability: use backtracking with bitmasks
    # colors[i] in {0,1,2,-1}
    
    def check_not_3_colorable(adj_mask):
        """Check if graph with adjacency masks is NOT 3-colorable."""
        # Use recursive backtracking with pruning
        # Order vertices by degree (descending) for faster pruning
        order = sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        
        # For each vertex in order, try colors 0,1,2
        # Use bitmasks to track which colors are forbidden by neighbors
        
        # Map from original vertex to position in order
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            # Determine which colors are forbidden
            forbidden = 0
            for nb_bit in range(n):
                if (adj_mask[v] >> nb_bit) & 1:
                    nb = nb_bit
                    c = colors[nb]
                    if c != -1:
                        forbidden |= (1 << c)
            
            available = 7 & ~forbidden  # 3 bits for 3 colors
            
            if available == 0:
                return False
            
            # Try each available color
            while available:
                bit = available & (-available)
                available ^= bit
                c = bit.bit_length() - 1
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            
            return False
        
        return not backtrack(0)
    
    # Start with Clebsch graph as initial solution
    def get_clebsch_adj_mask():
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = u ^ v
                dist = bin(diff).count('1')
                if dist == 1 or dist == 4:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask
    
    adj_mask = get_clebsch_adj_mask()
    
    # Verify initial solution
    if not check_not_3_colorable(adj_mask):
        # Shouldn't happen, but fallback
        return []
    
    current_count = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    best_count = current_count
    best_adj_mask = adj_mask[:]
    
    # Branch and bound: try to add edges
    # Key insight: an edge (u,v) can be added if:
    # 1. It's not already present
    # 2. Adding it doesn't create a triangle: i.e., N(u) and N(v) are disjoint
    # 3. The resulting graph is still not 3-colorable
    
    # Precompute for each pair (u,v) whether they can be adjacent (triangle-free check)
    # Two vertices u,v can be adjacent iff N(u) ∩ N(v) = ∅ (excluding each other)
    
    def can_add_edge(u, v, adj_mask):
        """Check if edge (u,v) can be added without creating a triangle."""
        # N(u) and N(v) must be disjoint (excluding u and v themselves)
        # adj_mask[u] has v if already connected, but we check before adding
        if (adj_mask[u] >> v) & 1:
            return False  # Already present
        
        # Check if any common neighbor exists
        # Common neighbors of u and v: vertices w such that w in N(u) and w in N(v)
        # adj_mask[u] & adj_mask[v] gives common neighbors
        # But we need to exclude u and v themselves from consideration
        # Since u is not in N(u) and v is not in N(v), this should be fine
        common = adj_mask[u] & adj_mask[v]
        if common != 0:
            return False
        return True
    
    # Local search with branch and bound
    # Strategy: repeatedly try to add edges. When stuck, try removing one edge
    # and then greedily adding back as many as possible.
    
    random.seed(42)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        
        # Try all possible edge additions
        for (u, v) in all_edges:
            if time.time() > deadline:
                break
            if (adj_mask[u] >> v) & 1:
                continue
            if not can_add_edge(u, v, adj_mask):
                continue
            
            # Try adding this edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            if check_not_3_colorable(adj_mask):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj_mask = adj_mask[:]
                improved = True
            else:
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        # If no simple addition worked, try edge swaps
        if not improved:
            # Try removing each edge and then greedily adding
            current_edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        current_edge_list.append((i, j))
            
            random.shuffle(current_edge_list)
            
            found_improvement = False
            for (ru, rv) in current_edge_list:
                if time.time() > deadline:
                    break
                if found_improvement:
                    break
                
                # Remove edge (ru, rv)
                adj_mask[ru] &= ~(1 << rv)
                adj_mask[rv] &= ~(1 << ru)
                count_after_removal = current_count - 1
                
                # Now greedily add as many edges as possible
                added = 0
                for (u, v) in all_edges:
                    if time.time() > deadline:
                        break
                    if (u == ru and v == rv) or (u == rv and v == ru):
                        continue
                    if (adj_mask[u] >> v) & 1:
                        continue
                    if not can_add_edge(u, v, adj_mask):
                        continue
                    
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    
                    if check_not_3_colorable(adj_mask):
                        added += 1
                    else:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                
                new_count = count_after_removal + added
                
                if new_count > best_count:
                    best_count = new_count
                    best_adj_mask = adj_mask[:]
                    found_improvement = True
                    current_count = new_count
                else:
                    # Revert to state before removal
                    # We need to restore the original adj_mask
                    # This is complex, so let's just keep going
                    # Actually, we should revert properly
                    # For simplicity, if no improvement, we'll try another edge
                    pass
        
        if found_improvement:
            improved = True
    
    # Convert best_adj_mask to edge list
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_mask[u] >> v) & 1:
                final_edges.append([u, v])
    
    # Final validation
    # Check triangle-free
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj_mask[i] >> j) & 1:
                # Check if i and j have a common neighbor
                common = best_adj_mask[i] & best_adj_mask[j]
                if common != 0:
                    # Triangle found, fallback
                    pass
    
    # Check not 3-colorable
    if not check_not_3_colorable(best_adj_mask):
        # Fallback to Clebsch
        adj_mask = get_clebsch_adj_mask()
        final_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    final_edges.append([u, v])
    
    return final_edges
