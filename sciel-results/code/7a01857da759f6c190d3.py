    def _is_3_colorable(adj, deadline=None):
        n = 16
        color = [-1] * n
        
        def backtrack(idx):
            if deadline and time.time() > deadline:
                return None
            
            if idx == n:
                return True
            
            # MRV: find uncolored vertex with fewest options
            # Since we are coloring in a fixed order or dynamic?
            # Dynamic MRV is better.
            
            min_opts = 4
            best_v = -1
            best_opts = 0
            
            for v in range(n):
                if color[v] == -1:
                    opts = 7
                    mask = adj[v]
                    while mask:
                        lb = mask & (-mask)
                        u = lb.bit_length() - 1
                        mask ^= lb
                        if color[u] != -1:
                            opts &= ~(1 << color[u])
                    cnt = bin(opts).count('1')
                    if cnt == 0:
                        return False
                    if cnt < min_opts:
                        min_opts = cnt
                        best_v = v
                        best_opts = opts
                        if cnt == 1:
                            break
            
            if best_v == -1:
                return True
            
            # Try colors
            c = 0
            while c < 3:
                if best_opts & (1 << c):
                    color[best_v] = c
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    color[best_v] = -1
                c += 1
            return False
        
        res = backtrack(0)
        if res is None:
            return True # Timeout, assume 3-colorable to be safe
        return res
    