"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_not_3_colorable_masks(masks, n):
    """Check if graph is NOT 3-colorable using bitmask-based backtracking with MRV."""
    # Precompute adjacency lists as bitmasks
    adj = masks[:]
    
    # Use iterative deepening with MRV heuristic
    uncolored = (1 << n) - 1
    color_of = [-1] * n
    
    def backtrack(uncolored_mask, color_assignment):
        if uncolored_mask == 0:
            return True
        
        # MRV: find uncolored vertex with fewest available colors
        best_v = -1
        best_avail = 4
        temp = uncolored_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            # Count forbidden colors from colored neighbors
            forbidden = 0
            m = adj[v]
            temp_m = m
            while temp_m:
                lsb_m = temp_m & (-temp_m)
                u = lsb_m.bit_length() - 1
                if color_assignment[u] != -1:
                    forbidden |= (1 << color_assignment[u])
                temp_m ^= lsb_m
            avail = 3 - bin(forbidden).count('1')
            if avail < best_avail:
                best_avail = avail
                best_v = v
                if avail == 0:
                    break
            temp ^= lsb
        
        if best_v == -1:
            return True
        
        # Try all valid colors for best_v
        forbidden = 0
        m = adj[best_v]
        temp_m = m
        while temp_m:
            lsb_m = temp_m & (-temp_m)
            u = lsb_m.bit_length() - 1
            if color_assignment[u] != -1:
                forbidden |= (1 << color_assignment[u])
            temp_m ^= lsb_m
        
        for c in range(3):
            if not (forbidden >> c) & 1:
                color_assignment[best_v] = c
                if backtrack(uncolored_mask & ~(1 << best_v), color_assignment):
                    return True
                color_assignment[best_v] = -1
        
        return False
    
    color_assignment = [-1] * n
    return not backtrack((1 << n) - 1, color_assignment)

def _get_edges_from_masks(masks, n):
    edges = []
    for u in range(n):
        m = masks[u]
        v = 0
        while m:
            if m & 1:
                if u < v:
                    edges.append([u, v])
            m >>= 1
            v += 1
    return edges

def _count_edges(masks):
    return sum(bin(m).count('1') for m in masks) // 2

def _greedy_add_edges(masks, n, deadline):
    """Try to greedily add edges while maintaining triangle-free and 4-chromatic."""
    current_count = _count_edges(masks)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if (masks[u] >> v) & 1:
                    continue
                if (masks[u] & masks[v]) != 0:
                    continue
                # Check if adding edge preserves 4-chromaticity
                new_masks = masks[:]
                new_masks[u] |= (1 << v)
                new_masks[v] |= (1 << u)
                if _is_not_3_colorable_masks(new_masks, n):
                    masks = new_masks
                    current_count += 1
                    improved = True
                    if time.time() > deadline:
                        return masks, current_count
    return masks, current_count

def _edge_swap_perturbation(masks, n, deadline):
    """Remove a random edge and try to add different edges."""
    edges = _get_edges_from_masks(masks, n)
    if not edges:
        return masks, _count_edges(masks)
    
    current_count = _count_edges(masks)
    u, v = random.choice(edges)
    masks[u] &= ~(1 << v)
    masks[v] &= ~(1 << u)
    current_count -= 1
    
    # Try to add different edges
    for u2 in range(n):
        for v2 in range(u2 + 1, n):
            if (masks[u2] >> v2) & 1:
                continue
            if (masks[u2] & masks[v2]) != 0:
                continue
            new_masks = masks[:]
            new_masks[u2] |= (1 << v2)
            new_masks[v2] |= (1 << u2)
            if _is_not_3_colorable_masks(new_masks, n):
                masks = new_masks
                current_count += 1
                if time.time() > deadline:
                    return masks, current_count
    
    return masks, current_count

def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    n = 16
    random.seed(42)
    
    def random_triangle_free_graph():
        masks = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if (masks[u] & masks[v]) == 0:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        return masks
    
    best_masks = [0] * n
    best_edge_count = 0
    
    # Start with a known 4-chromatic triangle-free graph: the Grötzsch graph (11 vertices)
    # embedded in 16 vertices, then extend
    groetzsch_edges = [
        [0,1],[0,2],[0,3],[0,4],[0,5],
        [1,2],[2,3],[3,4],[4,5],[5,1],
        [1,3],[2,4],[3,5],[4,1],[5,2],
        [6,7],[6,8],[6,9],[6,10],
        [7,8],[8,9],[9,10],[10,7],
        [7,9],[8,10],[9,7],[10,8],
    ]
    # Map 11 vertices to 16 vertices (use first 11)
    masks = [0] * n
    for u, v in groetzsch_edges:
        if u < 11 and v < 11:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
    
    # Verify it's 4-chromatic
    if _is_not_3_colorable_masks(masks, n):
        best_masks = masks[:]
        best_edge_count = _count_edges(masks)
    
    # Try multiple strategies
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if trial % 3 == 0:
            masks = random_triangle_free_graph()
        elif trial % 3 == 1:
            # Start from best known and perturb
            masks = best_masks[:]
            if _count_edges(masks) > 0:
                masks, _ = _edge_swap_perturbation(masks, n, deadline)
        else:
            # Start from Grötzsch and extend
            masks = best_masks[:] if best_edge_count > 0 else [0]*n
        
        # Greedy edge addition
        masks, current_count = _greedy_add_edges(masks, n, deadline)
        
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_masks = masks[:]
        
        # Edge swap perturbations to escape local optima
        if time.time() < deadline:
            for _ in range(5):
                if time.time() > deadline:
                    break
                masks, current_count = _edge_swap_perturbation(masks, n, deadline)
                masks, current_count = _greedy_add_edges(masks, n, deadline)
                
                if current_count > best_edge_count:
                    best_edge_count = current_count
                    best_masks = masks[:]
    
    # Final verification
    if best_edge_count == 0:
        return []
    
    if not _is_not_3_colorable_masks(best_masks, n):
        for trial in range(50):
            if time.time() > deadline:
                break
            masks = random_triangle_free_graph()
            if _is_not_3_colorable_masks(masks, n):
                best_masks = masks
                best_edge_count = _count_edges(masks)
                break
    
    return _get_edges_from_masks(best_masks, n)
# EVOLVE-BLOCK-END
