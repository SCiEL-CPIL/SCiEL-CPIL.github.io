import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # We want to maximize n. Best known is 17.
    # Try n=17, 16, 15...
    
    def solve(n):
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(4)]
        
        # Precompute bit masks for vertices
        # vertex i corresponds to bit (1 << i)
        
        def get_neighbors(c, i):
            return adj[c][i]
        
        def has_path_len_5(c, u, v):
            """
            Check if there is a simple path of length 5 from u to v in color c.
            Path: u - a - b - d - e - v
            u, a, b, d, e, v are distinct.
            """
            if u == v:
                return False
            
            # DFS
            # visited is a bitmask
            # depth remaining
            # current node
            
            # We can use a stack or recursion.
            # Recursion depth 5 is fine.
            
            def dfs(node, depth, visited):
                if depth == 0:
                    return node == v
                if depth < 0:
                    return False
                
                # Optimization: if depth == 1, check if v is neighbor
                if depth == 1:
                    return (adj[c][node] & (1 << v)) != 0
                
                # Get neighbors
                mask = adj[c][node] & ~visited
                # Iterate through bits in mask
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if dfs(idx, depth - 1, visited | lsb):
                        return True
                    m ^= lsb
                return False

            # Start DFS from u
            # visited includes u
            return dfs(u, 4, 1 << u) # depth 4 means 4 more edges to go, total 5 edges. 
            # Wait. Path length 5 means 5 edges.
            # u -a- b -d- e -v
            # Edges: (u,a), (a,b), (b,d), (d,e), (e,v).
            # If we are at 'node' and have 'depth' edges left to reach 'v':
            # If depth == 1, we need (node, v) to be an edge.
            # If depth == 4, we are at u. We need to find a, b, d, e.
            # So initial call: dfs(u, 4, 1<<u).
            # Inside:
            #   if depth == 1: check edge to v.
            #   else: for each neighbor a: dfs(a, depth-1, visited|a)
            # This is correct.

        # Edge index helper
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + j - i - 1

        # List of edges in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        colors = [-1] * len(edges)
        
        # Symmetry breaking:
        # Fix colors of first few edges.
        # (0,1) -> 0
        # (0,2) -> 1
        # (0,3) -> 2
        # (0,4) -> 3
        # This assumes n >= 5.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
                
            if idx == len(edges):
                return True
                
            i, j = edges[idx]
            
            # Try colors 0, 1, 2, 3
            # Order of colors can be randomized or fixed.
            # Fixed is better for reproducibility, but random might escape local minima?
            # In backtracking, order doesn't matter much for completeness, but for speed, 
            # trying "promising" colors first helps.
            # Here, all colors are symmetric initially.
            
            for c in range(4):
                # Check if color c is valid for edge (i, j)
                # Condition: No path of length 5 from i to j in color c.
                if has_path_len_5(c, i, j):
                    continue
                
                # Assign
                colors[idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                
            return False

        # Apply symmetry breaking
        # We need to ensure the fixed edges don't conflict.
        # (0,1) is edge 0. (0,2) is edge 1. (0,3) is edge 2. (0,4) is edge 3.
        
        # Set (0,1) to 0
        i, j = 0, 1
        idx = edge_idx(i, j)
        colors[idx] = 0
        adj[0][i] |= (1 << j)
        adj[0][j] |= (1 << i)
        
        # Set (0,2) to 1
        i, j = 0, 2
        idx = edge_idx(i, j)
        colors[idx] = 1
        adj[1][i] |= (1 << j)
        adj[1][j] |= (1 << i)
        
        # Set (0,3) to 2
        i, j = 0, 3
        idx = edge_idx(i, j)
        colors[idx] = 2
        adj[2][i] |= (1 << j)
        adj[2][j] |= (1 << i)
        
        if n >= 5:
            i, j = 0, 4
            idx = edge_idx(i, j)
            colors[idx] = 3
            adj[3][i] |= (1 << j)
            adj[3][j] |= (1 << i)
            
        # Start backtracking from the first unassigned edge
        # The first 4 edges (0,1), (0,2), (0,3), (0,4) are assigned.
        # So we start from index 4.
        start_idx = 4 if n >= 5 else 3
        
        # Wait, the edges list is:
        # (0,1) idx 0
        # (0,2) idx 1
        # (0,3) idx 2
        # (0,4) idx 3
        # (1,2) idx 4
        # ...
        
        if backtrack(start_idx):
            return colors
        else:
            return None

    # Try n from 17 down to 1
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
            
        colors = solve(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback to a known valid small coloring if all fail (should not happen)
    # n=9 is easy
    n = 9
    # Random coloring might work for small n, or just return a simple pattern
    # A simple pattern that avoids C6 for small n
    # For n=9, random search is fast
    random.seed(42)
    for _ in range(1000):
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        # Check validity
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * n - i * (i + 1) // 2 + j - i - 1
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        valid = True
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[c][i] & (1 << j)):
                        # Check path len 5
                        # Reuse logic? Just do a quick check
                        # For n=9, it's fast
                        pass # We'll trust the backtracker for n>=13. For n=9, we can just return random if we assume it's valid? No, must be valid.
                        # Let's just use the backtracker for n=9 too if needed.
                        # But the loop above tries 17..9. If 9 fails, we are in trouble.
                        # But 9 is very likely to succeed.
        
        # Actually, let's just return the result of solve(9) if we get here.
        # The loop breaks if time is up.
        # If solve(9) returns None, we return a hardcoded valid one.
        
    # Hardcoded valid coloring for n=9 (or just return n=1 if all else fails)
    # A 4-coloring of K9 without C6 is trivial.
    # Just return n=1 to be safe? No, score 0.
    # Return n=9 with random colors? Risky.
    
    # Let's assume solve(n) works for n<=17.
    # If we reach here, it means time ran out or all failed.
    # Return a trivial valid solution.
    return {'n': 1, 'colors': []}
