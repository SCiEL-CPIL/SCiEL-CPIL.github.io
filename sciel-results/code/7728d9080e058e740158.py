import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def has_c6_in_color(adj_bits, u, v, color):
        # adj_bits[color] is a list of integers, where bit i is 1 if vertex i is adjacent
        # Check path u-w1-w2-w3-v
        # u is fixed, v is fixed.
        
        # Neighbors of u
        n_u = adj_bits[color][u]
        
        # Iterate w1 in n_u
        # w1 != v
        w1_mask = n_u & ~(1 << v)
        
        while w1_mask:
            # Get lowest set bit
            lsb = w1_mask & -w1_mask
            w1 = lsb.bit_length() - 1
            w1_mask ^= lsb
            
            # Neighbors of w1
            n_w1 = adj_bits[color][w1]
            # Exclude u, w1, v
            # w2 must be neighbor of w1, not u, not w1 (self), not v
            w2_mask = n_w1 & ~(1 << u) & ~(1 << w1) & ~(1 << v)
            
            while w2_mask:
                lsb2 = w2_mask & -w2_mask
                w2 = lsb2.bit_length() - 1
                w2_mask ^= lsb2
                
                # Neighbors of w2
                n_w2 = adj_bits[color][w2]
                # Exclude u, w1, w2, v
                w3_mask = n_w2 & ~(1 << u) & ~(1 << w1) & ~(1 << w2) & ~(1 << v)
                
                while w3_mask:
                    lsb3 = w3_mask & -w3_mask
                    w3 = lsb3.bit_length() - 1
                    w3_mask ^= lsb3
                    
                    # Check if v is neighbor of w3
                    if adj_bits[color][w3] & (1 << v):
                        return True
                        
        return False

    def solve(n):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        
        colors = [-1] * (n * (n - 1) // 2)
        # adj_bits[color][vertex] = bitmask of neighbors
        adj_bits = [[0] * n for _ in range(4)]
        
        def bt(edge_idx):
            if time.time() > deadline:
                return None
            
            if edge_idx == n * (n - 1) // 2:
                return colors[:]
            
            u, v = pairs[edge_idx]
            
            # Symmetry breaking
            if u == 0:
                if v == 1:
                    c_list = [0]
                elif v == 2:
                    c_list = [1]
                elif v == 3:
                    c_list = [2]
                elif v == 4:
                    c_list = [3]
                else:
                    c_list = [0, 1, 2, 3]
            else:
                c_list = [0, 1, 2, 3]
                
            for c in c_list:
                if has_c6_in_color(adj_bits, u, v, c):
                    continue
                
                colors[edge_idx] = c
                adj_bits[c][u] |= (1 << v)
                adj_bits[c][v] |= (1 << u)
                
                res = bt(edge_idx + 1)
                if res:
                    return res
                
                colors[edge_idx] = -1
                adj_bits[c][u] &= ~(1 << v)
                adj_bits[c][v] &= ~(1 << u)
                
            return None
        
        return bt(0)

    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for n in range(11, 14):
        if time.time() > deadline:
            break
        res = solve(n)
        if res:
            best_n = n
            best_colors = res

    return {'n': best_n, 'colors': best_colors}
