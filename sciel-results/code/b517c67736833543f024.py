        def creates_c6(u, v):
            # Check if there is a simple path of length 5 from u to v
            # S_k: vertices reachable from u by simple path of length k
            S_prev = 1 << u
            visited = S_prev
            
            for k in range(1, 6):
                # Compute neighbors of S_prev
                S_curr = 0
                temp = S_prev
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    S_curr |= adj[idx]
                    temp ^= lsb
                
                # Exclude visited
                S_curr &= ~visited
                visited |= S_curr
                
                if k == 5:
                    return (S_curr >> v) & 1
                else:
                    # If S_curr is empty, no longer paths possible
                    if S_curr == 0:
                        return False
                    S_prev = S_curr
            return False
        