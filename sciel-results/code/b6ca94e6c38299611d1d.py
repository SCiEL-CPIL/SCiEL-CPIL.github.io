import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 18
    
    # Precompute all subsets of size 12 as bitmasks
    subsets_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)
    
    def is_triangle_free(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True
    
    def find_all_triangle_free_12(adj):
        """Return list of masks of size-12 subsets that are triangle-free."""
        result = []
        for mask in subsets_12:
            if is_triangle_free(mask, adj):
                result.append(mask)
        return result
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True
    
    def generate_random_max_k4_free():
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    # Seed: complement of K6,6,6 (i.e., 3 disjoint K6's)
    # This has triangle-free induced subgraph of size 12 (one part from two K6's)
    seed_adj = [0] * n
    for i in range(6):
        for j in range(i+1, 6):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(i+1, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(12, 18):
        for j in range(i+1, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    
    best_edges = adj_to_edges(seed_adj)
    best_score = 12
    
    def count_triangles_in_mask(mask, adj):
        """Count triangles in the induced subgraph of mask."""
        count = 0
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    count += 1
        return count // 3  # Each triangle counted 3 times
    
    def edge_impact_score(adj, u, v, tf_masks):
        """Score how many triangle-free 12-subsets would be fixed by adding edge (u,v)."""
        if adj[u] & (1 << v):
            return 0  # Already an edge
        if not can_add_edge(adj, u, v):
            return 0  # Would create K4
        
        # Count how many tf_masks contain both u and v
        impact = 0
        for mask in tf_masks:
            if (mask & (1 << u)) and (mask & (1 << v)):
                impact += 1
        return impact
    
    def dpll_solve(adj, max_depth=50):
        """DPLL-style search to eliminate all triangle-free 12-subsets."""
        adj = list(adj)
        depth = 0
        max_depth = max_depth
        
        while depth < max_depth:
            if time.time() > deadline - 3:
                break
            
            tf_masks = find_all_triangle_free_12(adj)
            if not tf_masks:
                return adj
            
            # Find the best edge to add
            best_edge = None
            best_score = -1
            best_impact = 0
            
            # Get all non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
            
            # Score each non-edge
            for u, v in non_edges:
                if can_add_edge(adj, u, v):
                    impact = edge_impact_score(adj, u, v, tf_masks)
                    if impact > best_impact or (impact == best_impact and random.random() < 0.3):
                        best_impact = impact
                        best_edge = (u, v)
            
            if best_edge is None:
                # No valid edge to add - try edge removal
                # Find edges that are in many tf_masks
                edge_in_tf_count = {}
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            count = 0
                            for mask in tf_masks:
                                if (mask & (1 << u)) and (mask & (1 << v)):
                                    count += 1
                            edge_in_tf_count[(u, v)] = count
                
                # Remove an edge that's in many tf_masks to allow future additions
                if edge_in_tf_count:
                    # Find edge with highest count
                    max_count = max(edge_in_tf_count.values())
                    candidates = [e for e, c in edge_in_tf_count.items() if c == max_count]
                    u, v = random.choice(candidates)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    depth += 1
                    continue
                else:
                    break
            
            # Add the best edge
            u, v = best_edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            depth += 1
        
        return adj
    
    def local_search_improved(adj):
        """Improved local search with DPLL-style edge selection."""
        adj = list(adj)
        max_iter = 100
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            tf_masks = find_all_triangle_free_12(adj)
            if not tf_masks:
                return adj
            
            # Use DPLL-style selection
            adj = dpll_solve(adj, max_depth=20)
            
            tf_final = find_all_triangle_free_12(adj)
            if not tf_final:
                return adj
            
            # If still stuck, try random perturbations
            if random.random() < 0.3:
                # Remove a random edge
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append((u, v))
                if edges:
                    u, v = random.choice(edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return adj
    
    # Phase 1: Try random graphs with improved local search
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        adj = generate_random_max_k4_free()
        
        # Quick check: does it have any triangle-free 12-subset?
        tf_count = 0
        for mask in subsets_12:
            if is_triangle_free(mask, adj):
                tf_count += 1
                if tf_count > 30:  # Too many, skip
                    break
        
        if tf_count == 0:
            return adj_to_edges(adj)
        
        if tf_count <= 30:
            # Promising: run improved local search
            adj_ls = local_search_improved(adj)
            tf_final = find_all_triangle_free_12(adj_ls)
            if not tf_final:
                return adj_to_edges(adj_ls)
    
    # Phase 2: Try improved local search from seed
    adj_seed = list(seed_adj)
    adj_ls = local_search_improved(adj_seed)
    tf_final = find_all_triangle_free_12(adj_ls)
    if not tf_final:
        return adj_to_edges(adj_ls)
    
    # Phase 3: More random restarts with improved local search
    while time.time() < deadline - 2:
        adj = generate_random_max_k4_free()
        adj_ls = local_search_improved(adj)
        tf_final = find_all_triangle_free_12(adj_ls)
        if not tf_final:
            return adj_to_edges(adj_ls)
    
    return best_edges

# EVOLVE-BLOCK-END
