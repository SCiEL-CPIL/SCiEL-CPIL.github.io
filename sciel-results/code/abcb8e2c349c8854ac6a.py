import time
import random

DEADLINE = 110  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    n: number of vertices
    adj_masks: list of int bitmasks representing adjacency
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0
    
    # Iterate over all edges (i, j) with i < j
    for i in range(n):
        mi = adj_masks[i]
        # Only consider j > i
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb
            
            # Common neighbors of i and j
            common = mi & adj_masks[j]
            
            # Iterate k in common
            max_deg_k = -1
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                d = deg[k]
                if d > max_deg_k:
                    max_deg_k = d
                mask_k ^= lsb_k
            
            if max_deg_k >= 0:
                s = deg[i] + deg[j] + max_deg_k
                if s > max_sum:
                    max_sum = s
                    
    return max_sum

def _construct_edges_from_masks(adj_masks, n):
    edges = []
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            edges.append([i, j])
            mask_j_greater ^= lsb
    return edges

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    # Initial state: K_{8,8} + one internal edge (0,1)
    adj_masks = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
    adj_masks[0] |= (1 << 1)
    adj_masks[1] |= (1 << 0)
    
    current_score = _compute_score(n, adj_masks)
    best_score = current_score
    best_adj_masks = list(adj_masks)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Use sets for O(1) lookup and removal
    current_edge_set = set()
    non_edge_set = set()
    
    for e in all_edges:
        u, v = e
        if (adj_masks[u] >> v) & 1:
            current_edge_set.add(e)
        else:
            non_edge_set.add(e)
            
    random.seed(42)
    
    max_iterations = 50000
    iterations = 0
    stuck_count = 0
    
    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1
        
        # Strategy:
        # 1. Try to improve by targeting the worst triangle.
        # 2. If that fails, try random swaps.
        
        improved = False
        
        # Find worst triangle(s)
        deg = [bin(mask).count('1') for mask in adj_masks]
        worst_sum = 0
        worst_triangle = None
        
        for i in range(n):
            mi = adj_masks[i]
            mask_j_greater = mi & ~((1 << (i + 1)) - 1)
            while mask_j_greater:
                lsb = mask_j_greater & -mask_j_greater
                j = lsb.bit_length() - 1
                mask_j_greater ^= lsb
                common = mi & adj_masks[j]
                mask_k = common
                while mask_k:
                    lsb_k = mask_k & -mask_k
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    s = deg[i] + deg[j] + deg[k]
                    if s > worst_sum:
                        worst_sum = s
                        worst_triangle = (i, j, k)
                        
        if worst_triangle:
            i, j, k = worst_triangle
            # Try removing each edge of the worst triangle
            edges_to_try = [(i, j), (j, k), (i, k)]
            random.shuffle(edges_to_try)
            
            for rem_u, rem_v in edges_to_try:
                # Remove edge
                adj_masks[rem_u] ^= (1 << rem_v)
                adj_masks[rem_v] ^= (1 << rem_u)
                
                # Update sets
                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                current_edge_set.discard(rem_edge)
                non_edge_set.add(rem_edge)
                
                # Try adding random non-edges
                attempts = 0
                found_improvement = False
                while attempts < 8:
                    if not non_edge_set:
                        break
                    attempts += 1
                    add_u, add_v = random.choice(list(non_edge_set))
                    add_edge = (min(add_u, add_v), max(add_u, add_v))
                    
                    # Add edge
                    adj_masks[add_u] ^= (1 << add_v)
                    adj_masks[add_v] ^= (1 << add_u)
                    
                    # Update sets
                    non_edge_set.discard(add_edge)
                    current_edge_set.add(add_edge)
                    
                    new_score = _compute_score(n, adj_masks)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj_masks = list(adj_masks)
                        break
                    else:
                        # Revert add
                        adj_masks[add_u] ^= (1 << add_v)
                        adj_masks[add_v] ^= (1 << add_u)
                        
                        # Revert sets
                        current_edge_set.discard(add_edge)
                        non_edge_set.add(add_edge)
                
                if found_improvement:
                    break
                else:
                    # Revert removal
                    adj_masks[rem_u] ^= (1 << rem_v)
                    adj_masks[rem_v] ^= (1 << rem_u)
                    
                    # Revert sets
                    non_edge_set.discard(rem_edge)
                    current_edge_set.add(rem_edge)
                    
        if not improved:
            # Try a random swap
            if non_edge_set and current_edge_set:
                rem_edge = random.choice(list(current_edge_set))
                add_edge = random.choice(list(non_edge_set))
                
                rem_u, rem_v = rem_edge
                add_u, add_v = add_edge
                
                # Perform swap
                adj_masks[rem_u] ^= (1 << rem_v)
                adj_masks[rem_v] ^= (1 << rem_u)
                current_edge_set.discard(rem_edge)
                non_edge_set.add(rem_edge)
                
                adj_masks[add_u] ^= (1 << add_v)
                adj_masks[add_v] ^= (1 << add_u)
                non_edge_set.discard(add_edge)
                current_edge_set.add(add_edge)
                
                new_score = _compute_score(n, adj_masks)
                
                # Accept if better, or with small probability if equal (to explore)
                if new_score < current_score or (new_score == current_score and random.random() < 0.15):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj_masks = list(adj_masks)
                    stuck_count = 0
                else:
                    # Revert
                    adj_masks[rem_u] ^= (1 << rem_v)
                    adj_masks[rem_v] ^= (1 << rem_u)
                    non_edge_set.discard(rem_edge)
                    current_edge_set.add(rem_edge)
                    
                    adj_masks[add_u] ^= (1 << add_v)
                    adj_masks[add_v] ^= (1 << add_u)
                    current_edge_set.discard(add_edge)
                    non_edge_set.add(add_edge)
                    stuck_count += 1
                    
                # Perturbation if stuck
                if stuck_count > 80:
                    for _ in range(4):
                        if non_edge_set and current_edge_set:
                            r_edge = random.choice(list(current_edge_set))
                            a_edge = random.choice(list(non_edge_set))
                            
                            r_u, r_v = r_edge
                            a_u, a_v = a_edge
                            
                            adj_masks[r_u] ^= (1 << r_v)
                            adj_masks[r_v] ^= (1 << r_u)
                            current_edge_set.discard(r_edge)
                            non_edge_set.add(r_edge)
                            
                            adj_masks[a_u] ^= (1 << a_v)
                            adj_masks[a_v] ^= (1 << a_u)
                            non_edge_set.discard(a_edge)
                            current_edge_set.add(a_edge)
                            
                    current_score = _compute_score(n, adj_masks)
                    stuck_count = 0

    best_edges = _construct_edges_from_masks(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
