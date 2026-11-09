import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Try to find a valid coloring for n=29
    for n in range(29, 16, -1):
        if time.time() > deadline - 5:
            break
        result = search_coloring(n, start_time, deadline)
        if result is not None:
            return result
            
    # Fallback to seed if search fails
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

def search_coloring(n, start_time, deadline):
    # adj[c][v] is a bitmask of neighbors of v in color c
    adj = [[0] * n for _ in range(2)]
    
    # Edge order: lexicographic (i, j) with i < j
    # We fix some edges for symmetry breaking
    # Fix (0, 1) = 0
    # Fix (0, 2) = 0
    # Fix (0, 3) = 0
    
    def set_edge(i, j, c):
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
        adj[1-c][i] &= ~(1 << j)
        adj[1-c][j] &= ~(1 << i)
        
    def get_edge(i, j):
        if (adj[0][i] >> j) & 1:
            return 0
        else:
            return 1
            
    def creates_k222(u, v, c):
        # Check if adding edge (u, v) with color c creates a K2,2,2
        # u, v are the new endpoints.
        # We assume the edge is already set in adj.
        
        # Candidates for u' in N_c(v) \ {u}
        # (u, u') must be 1-c
        mask_v = adj[c][v]
        # u_prime must be in mask_v
        # and NOT in adj[c][u] (so that (u, u') is 1-c)
        # and u_prime != u (u is not in adj[c][u] anyway)
        
        candidates_u_prime = mask_v & ~adj[c][u]
        
        # Iterate over set bits in candidates_u_prime
        m = candidates_u_prime
        while m:
            lsb = m & -m
            u_prime = lsb.bit_length() - 1
            m ^= lsb
            
            # Find W = N_c(u) & N_c(u_prime) & N_c(v)
            mask_w = adj[c][u] & adj[c][u_prime] & adj[c][v]
            
            # Iterate pairs w, x in mask_w
            # We need (w, x) to be 1-c
            # And we need v' such that:
            # v' in N_c(u) & N_c(u_prime) & N_c(w) & N_c(x)
            # v' != v
            # (v', v) is 1-c  => v' not in adj[c][v]
            
            # Precompute base mask for v'
            mask_vp_base = adj[c][u] & adj[c][u_prime]
            
            # Iterate pairs in mask_w
            mw = mask_w
            while mw:
                lsb_w = mw & -mw
                w = lsb_w.bit_length() - 1
                mw ^= lsb_w
                
                # x must be in mask_w and x > w to avoid duplicates? No, just x != w
                # But we can just iterate all x in mask_w \ {w}
                # Actually, we can just check if there exists ANY x in mask_w \ {w}
                # such that conditions hold.
                
                # Check (w, x) is 1-c
                # If (w, x) is c, then x is in adj[c][w]
                
                # Let's iterate x
                mx = mask_w & ~(1 << w)
                while mx:
                    lsb_x = mx & -mx
                    x = lsb_x.bit_length() - 1
                    mx ^= lsb_x
                    
                    # Check (w, x) is 1-c
                    if (adj[c][w] >> x) & 1:
                        continue
                        
                    # Find v'
                    mask_vp = mask_vp_base & adj[c][w] & adj[c][x]
                    # v' != v
                    mask_vp &= ~(1 << v)
                    # (v', v) is 1-c => v' not in adj[c][v]
                    mask_vp &= ~adj[c][v]
                    
                    if mask_vp != 0:
                        return True
                        
        return False

    # Backtracking
    # We process edges in lexicographic order
    # To speed up, we can use a stack or recursion
    # Recursion depth is n*(n-1)/2 which is ~400 for n=29.
    # Python recursion limit might be an issue, but 400 is fine.
    
    # Symmetry breaking:
    # Fix (0,1)=0, (0,2)=0, (0,3)=0
    # Also, we can assume the first edge (0,1) is 0.
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
            
    # Pre-assign symmetry breaking edges
    # (0,1)=0, (0,2)=0, (0,3)=0
    # We need to be careful if n < 4
    if n >= 2: set_edge(0, 1, 0)
    if n >= 3: set_edge(0, 2, 0)
    if n >= 4: set_edge(0, 3, 0)
    
    # Find indices of these edges in `edges` to skip them
    # Or just start from the next unassigned edge.
    # Let's find the index of (0,3) in edges
    start_idx = 0
    if n >= 4:
        # (0,3) is at index 3 in 0,1; 0,2; 0,3
        start_idx = 3
    elif n == 3:
        start_idx = 2
    elif n == 2:
        start_idx = 1
    else:
        start_idx = 0
        
    def backtrack(idx):
        if time.time() > deadline:
            return None
            
        if idx == len(edges):
            return True
            
        u, v = edges[idx]
        
        # Try color 0
        set_edge(u, v, 0)
        if not creates_k222(u, v, 0):
            res = backtrack(idx + 1)
            if res is not None:
                return res
        set_edge(u, v, 1) # Undo by setting to 1? No, set_edge overwrites.
        # We need to restore state.
        # Since set_edge modifies adj, we should save/restore or use a stack.
        # For simplicity, let's use a global state and undo.
        # Actually, set_edge(u,v,1) will set the edge to 1.
        # But we need to try 1 only if 0 failed or was not tried.
        # The logic above tries 0, and if it fails, it sets to 1 and tries 1.
        # But if 0 succeeded, we return.
        # If 0 failed, we set to 1 and check.
        
        # Wait, the code above:
        # set_edge(u, v, 0)
        # if not creates...:
        #    res = backtrack...
        #    if res: return res
        # set_edge(u, v, 1)
        # if not creates...:
        #    res = backtrack...
        #    if res: return res
        # return None
        
        # This is correct.
        
        if not creates_k222(u, v, 1):
            res = backtrack(idx + 1)
            if res is not None:
                return res
        return None
        
    # Run backtrack
    # To avoid recursion depth issues, we can use iterative, but 400 is fine.
    # However, Python is slow. 150s might not be enough for n=29 with this simple pruning.
    # But let's try.
    
    # Optimization: Randomize edge order? No, lexicographic is good for symmetry breaking.
    
    # If n=29 is too slow, it will hit the deadline.
    # The deadline check is in backtrack.
    
    # We need to capture the result.
    # If backtrack returns True, we have a valid coloring.
    # We need to extract the colors.
    
    # But backtrack returns True/None.
    # We need to store the final state.
    # The `adj` matrix is modified in place.
    # If it returns True, `adj` holds the solution.
    
    # However, if it returns None, `adj` is in some partial state.
    # We just discard it.
    
    # Let's run it
    # To prevent stack overflow, we increase recursion limit
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    success = backtrack(start_idx)
    
    sys.setrecursionlimit(old_limit)
    
    if success:
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(get_edge(i, j))
        return {'n': n, 'colors': colors}
        
    return None
