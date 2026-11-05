import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 112.0  # safety margin before 120s

    def time_exceeded():
        return time.time() - start_time > deadline

    # Paley(29) construction
    q = 29
    squares_29 = set()
    for x in range(1, q):
        squares_29.add((x * x) % q)

    def build_paley_colors(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares_29:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def check_k222_fast(n, colors):
        """
        Check for K_{2,2,2} in either color class using bitmasks.
        Returns True if either color class contains K_{2,2,2}.
        """
        # Build adjacency as list of bitmasks for each color
        adj = [[0] * n, [0] * n]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        def has_k222(adj_color):
            # Precompute non-edge pairs
            non_edge_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_color[i] & (1 << j)):
                        non_edge_pairs.append((i, j))
            
            if len(non_edge_pairs) < 3:
                return False
            
            # For each non-edge pair (i,j) as potential group A:
            for (i, j) in non_edge_pairs:
                # Common neighbors of i and j
                cn_mask = adj_color[i] & adj_color[j]
                cn_mask &= ~(1 << i) & ~(1 << j)
                
                # Get list of common neighbors
                cn_list = []
                mask = cn_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    cn_list.append(v)
                    mask ^= lsb
                
                if len(cn_list) < 4:
                    continue
                
                # Find non-edge pairs (k,l) in cn_list
                for a_idx in range(len(cn_list)):
                    k = cn_list[a_idx]
                    for b_idx in range(a_idx + 1, len(cn_list)):
                        l = cn_list[b_idx]
                        if adj_color[k] & (1 << l):
                            continue
                        
                        # Common neighbors of i, j, k, l
                        cn2_mask = adj_color[i] & adj_color[j] & adj_color[k] & adj_color[l]
                        cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                        
                        if cn2_mask == 0:
                            continue
                        
                        # Check if cn2_mask contains a non-edge pair
                        # Get list
                        cn2_list = []
                        mask2 = cn2_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            cn2_list.append(v2)
                            mask2 ^= lsb2
                        
                        if len(cn2_list) < 2:
                            continue
                        
                        for c_idx in range(len(cn2_list)):
                            m = cn2_list[c_idx]
                            for d_idx in range(c_idx + 1, len(cn2_list)):
                                o = cn2_list[d_idx]
                                if not (adj_color[m] & (1 << o)):
                                    return True
            
            return False
        
        return has_k222(adj[0]) or has_k222(adj[1])

    # Verify Paley(29) is valid
    paley_valid = not check_k222_fast(q, colors_29)
    
    if paley_valid:
        # Try to extend to n=30 via local search
        best_n = q
        best_colors = colors_29[:]
        
        import random
        random.seed(42)
        
        max_attempts = 300
        for attempt in range(max_attempts):
            if time_exceeded():
                break
            
            # Create initial coloring for n=30
            colors_30 = colors_29[:]
            
            # Initialize edges from vertex 29 to vertices 0..28
            for i in range(29):
                colors_30.append(random.randint(0, 1))
            
            # Local search
            for step in range(500):
                if time_exceeded():
                    break
                if not check_k222_fast(30, colors_30):
                    return {'n': 30, 'colors': colors_30}
                
                # Flip a random edge
                edge_idx = random.randint(0, len(colors_30) - 1)
                colors_30[edge_idx] = 1 - colors_30[edge_idx]
        
        return {'n': best_n, 'colors': best_colors}
    else:
        # Paley(29) is NOT valid - local search on n=29
        import random
        random.seed(42)
        
        colors = colors_29[:]
        for iteration in range(10000):
            if time_exceeded():
                break
            
            if not check_k222_fast(29, colors):
                return {'n': 29, 'colors': colors}
            
            edge_idx = random.randint(0, len(colors) - 1)
            colors[edge_idx] = 1 - colors[edge_idx]
        
        return {'n': 29, 'colors': colors}
