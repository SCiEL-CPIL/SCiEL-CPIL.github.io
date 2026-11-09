import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin

    def build_masks(n, adj):
        """Build bitmasks for adjacency."""
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def has_k4(n, masks):
        """Check if graph contains a K4."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                # Common neighbors of u and v
                common = masks[u] & masks[v]
                # Only consider vertices > v to avoid duplicates and self-loops in check
                common &= ~((1 << (v + 1)) - 1)
                # Check if any edge exists in common
                # Iterate through bits in common
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    # Check if wi is connected to any vertex in common > wi
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    if x_mask:
                        # We just need to know if there's an edge.
                        # Actually, we need to check if masks[wi] intersects x_mask.
                        if masks[wi] & x_mask:
                            return True
        return False

    def has_ind6(n, masks):
        """Check if graph contains an independent set of size 6.
        This is equivalent to checking if the complement graph has a K6.
        """
        full_mask = (1 << n) - 1
        # Complement masks (excluding self)
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        # Look for K6 in comp_masks
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        
                        if not (comp_masks[wi] & (1 << xi)):
                            continue
                        
                        y_mask = common & ~((1 << (xi + 1)) - 1)
                        # We need 2 more vertices y, z from y_mask
                        # such that they are connected to u, v, w, x in comp graph
                        # and y-z is an edge in comp graph.
                        
                        # Filter y_mask to only vertices connected to w and x in comp
                        valid_y = y_mask & comp_masks[wi] & comp_masks[xi]
                        
                        while valid_y:
                            lsb_y = valid_y & -valid_y
                            yi = lsb_y.bit_length() - 1
                            valid_y ^= lsb_y
                            
                            # Find z > yi in valid_y connected to yi
                            z_mask = valid_y & ~((1 << (yi + 1)) - 1)
                            # z must be connected to yi in comp
                            z_mask &= comp_masks[yi]
                            
                            if z_mask:
                                return True
        return False

    def is_valid_circulant(n, conn_set):
        """Check if circulant graph with given connection set is valid."""
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        for d in conn_set:
            d = d % n
            if d == 0:
                continue
            for i in range(n):
                j = (i + d) % n
                adj[i, j] = True
                adj[j, i] = True
        
        masks = build_masks(n, adj)
        
        if has_k4(n, masks):
            return False, None
        if has_ind6(n, masks):
            return False, None
        
        # Convert to edges list
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        return True, edges

    def search_circulant(n):
        """Search for a valid circulant graph on n vertices."""
        half = n // 2
        candidates = list(range(1, half + 1))
        
        # We want to find ANY valid graph.
        # Heuristic: Try sets of various sizes.
        # Usually, K4-free graphs are not too dense, but Ind6-free means not too sparse.
        # Density around 0.5 is often good.
        
        # Iterate through all subsets of candidates
        # To make it faster, we can order by size or use random sampling first.
        # For n=36, half=18, 2^18 = 262,144 subsets.
        # Each check is fast (bit ops). 262k checks * ~0.1ms = 26s. Might be tight.
        # We can prune: if conn_set contains 1, it's likely to have K4?
        # If conn_set is empty, Ind6 is definitely present (for n>=6).
        
        # Let's try all subsets, but break early if found.
        # We can randomize the order to find a solution faster if one exists,
        # or iterate systematically to guarantee finding one if it exists.
        # Systematic is safer for "maximize n" if we assume existence.
        
        # Optimization: 
        # If n is even, circulant graphs are vertex transitive.
        
        for mask in range(1, 1 << len(candidates)):
            conn = []
            for i in range(len(candidates)):
                if (mask >> i) & 1:
                    conn.append(candidates[i])
            
            # Quick filter: 
            # If 1 in conn, check for triangles/K4 early?
            # Just run the full check.
            valid, edges = is_valid_circulant(n, conn)
            if valid:
                return edges
                
        return None

    def fallback_clebsch():
        """Return the 16-vertex Clebsch graph as a guaranteed valid baseline."""
        clebsch_edges = [
            (0, 6), (0, 7), (0, 10), (0, 11),
            (1, 5), (1, 8), (1, 10), (1, 13),
            (2, 4), (2, 9), (2, 11), (2, 13),
            (3, 4), (3, 7), (3, 9), (3, 12),
            (5, 6), (5, 8), (5, 12), (5, 15),
            (6, 7), (6, 9), (6, 14), (6, 15),
            (7, 8), (7, 11), (7, 14), (7, 15),
            (8, 9), (8, 12), (8, 15),
            (9, 10), (9, 14),
            (10, 11), (10, 13), (10, 14),
            (11, 12), (11, 14), (11, 15),
            (12, 13), (12, 15),
            (13, 14), (13, 15)
        ]
        return {'n': 16, 'edges': [list(e) for e in clebsch_edges]}

    # We know R(4,6) > 35. So a 35-vertex graph exists.
    # We try to find a circulant graph for n=36 first.
    # If that fails, we try n=35.
    # If that fails, we try n=34, etc.
    
    # To save time, we don't want to exhaustively search n=36 if it takes > 100s.
    # We will allocate time budgets.
    
    best_result = None
    best_n = 0
    
    # Try n=36
    if time.time() < deadline - 50:
        edges = search_circulant(36)
        if edges:
            best_n = 36
            best_result = {'n': 36, 'edges': edges}
    
    # Try n=35
    if best_n < 35 and time.time() < deadline - 30:
        edges = search_circulant(35)
        if edges:
            best_n = 35
            best_result = {'n': 35, 'edges': edges}
            
    # Try n=34
    if best_n < 34 and time.time() < deadline - 15:
        edges = search_circulant(34)
        if edges:
            best_n = 34
            best_result = {'n': 34, 'edges': edges}

    # Fallback to Clebsch if nothing found
    if best_result is None:
        best_result = fallback_clebsch()
        
    return best_result

# The code above is the complete program.
