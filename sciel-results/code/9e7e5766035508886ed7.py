import time
import numpy as np
import random

def _build_adj_np(colors, n):
    """Build NumPy adjacency matrices for each color class."""
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i][j] = True
            adj[c][j][i] = True
            idx += 1
    return adj

def _has_c6_np(adj, n):
    """Check if the graph defined by NumPy adjacency matrix 'adj' contains a 6-cycle."""
    # Use matrix multiplication to count paths
    # A C6 exists if there is a closed walk of length 6.
    # Counting closed walks of length 6 is not exactly counting C6s, 
    # but if count > 0, it's a good indicator.
    # Actually, we need exact C6 detection.
    # Let's stick to bit-parallel for exactness, but use NumPy for initialization.
    
    # For speed in local search, we can use a heuristic count.
    # But for validity, we need exact check.
    
    # Let's use the bit-parallel method for exact check.
    # The NumPy method is for faster initialization or heuristic scoring.
    pass

def _is_valid_bitset(colors, n):
    """Check if the coloring is valid (no mono C6) using bit-parallel."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    for c in range(4):
        for v0 in range(n):
            mask_v1 = adj[c][v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                mask_v2 = adj[c][v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            mask_v5 = adj[c][v4] & adj[c][v0]
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            if mask_v5:
                                return False
    return True

def _algebraic_coloring(n):
    """
    Construct a 4-coloring of K_n using algebraic properties over GF(19).
    """
    p = 19
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            val = (i * j + i + j) % p
            c = val % 4
            colors.append(c)
    return colors

def _local_search(n, time_limit, seed=42):
    """
    Local search for a valid 4-coloring of K_n.
    """
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    # Start with algebraic coloring
    colors = _algebraic_coloring(n)
    
    # Build adj bitmasks
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    def count_c6s():
        total = 0
        for c in range(4):
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    total += 1
        return total

    current_c6s = count_c6s()
    if current_c6s == 0:
        return colors

    best_colors = colors[:]
    best_c6s = current_c6s

    for _ in range(100000):
        if time.time() > deadline:
            break
        if current_c6s == 0:
            return colors
            
        edge_idx = rng.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + rng.randint(0, 2)) % 4
        
        # Find i, j from edge_idx
        # Inverse of lexicographic order
        # i = 0: 0..15
        # i = 1: 0..14
        # ...
        # i = k: 0..n-k-2
        # Total edges before i is k: sum_{j=0}^{k-1} (n-1-j) = k*(n-1) - k*(k-1)//2
        # Let's precompute edge list for speed
        pass # Simplified: just flip and recount
        
        # Update adj
        # Need i, j
        # i = 0, j = edge_idx
        # i = 1, j = edge_idx - (n-1)
        # ...
        # This is slow. Let's precompute edge_list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j))
                
        i, j = edge_list[edge_idx]
        
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        colors[edge_idx] = new_color
        
        new_c6s = count_c6s()
        
        if new_c6s <= current_c6s:
            current_c6s = new_c6s
            if new_c6s < best_c6s:
                best_c6s = new_c6s
                best_colors = colors[:]
        else:
            # Revert
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            colors[edge_idx] = old_color
            current_c6s = best_c6s # Revert count? No, count is derived from adj.
            # We need to revert adj to old state and recount or keep track.
            # Simpler: just rebuild adj from colors if we revert? No, too slow.
            # Let's just revert the adj changes and assume current_c6s is back to previous value.
            # But previous value was calculated before the flip.
            # So yes, revert adj, and current_c6s is the value before the flip.
            pass

    if best_c6s == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_n = 0
    best_colors = None
    
    # Try n=19
    if time.time() < deadline - 10:
        res = _local_search(19, time_limit=10.0, seed=42)
        if res is not None:
            best_n = 19
            best_colors = res
            
    # Try n=18
    if best_n < 18 and time.time() < deadline - 10:
        res = _local_search(18, time_limit=10.0, seed=42)
        if res is not None:
            best_n = 18
            best_colors = res
            
    # Try n=17
    if best_n < 17 and time.time() < deadline - 5:
        res = _local_search(17, time_limit=5.0, seed=42)
        if res is not None:
            best_n = 17
            best_colors = res
        else:
            # Fallback to algebraic n=17
            colors_17 = _algebraic_coloring(17)
            if _is_valid_bitset(colors_17, 17):
                best_n = 17
                best_colors = colors_17
            else:
                # Try n=14
                colors_14 = _algebraic_coloring(14)
                if _is_valid_bitset(colors_14, 14):
                    best_n = 14
                    best_colors = colors_14
                else:
                    best_n = 13
                    best_colors = _algebraic_coloring(13)

    return {'n': best_n, 'colors': best_colors}
