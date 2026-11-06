import time
import sys

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def attempt(n, time_budget):
        local_deadline = start_time + time_budget
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        colors = [-1] * num_edges
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        
        # Precompute edge indices for quick lookup if needed, 
        # but we are assigning in order, so we just need to check validity.
        
        def has_path_len_5(c, u, v):
            # Check if there is a simple path of length 5 from u to v in color c
            # u - x1 - x2 - x3 - x4 - v
            # x1 in N(u), x2 in N(x1), x3 in N(x2), x4 in N(x3), v in N(x4)
            # All vertices distinct.
            
            # Optimization: 
            # If u and v are not connected by any path, return False.
            # We can do a BFS up to depth 5.
            
            # To make it faster, we can iterate over x1, x2, x3, x4.
            # But nested loops can be slow.
            
            # Let's use a recursive DFS with pruning.
            # State: (current_node, depth_remaining, visited_mask)
            
            def dfs(cur, depth, visited_mask):
                if depth == 0:
                    return cur == v
                if depth == 1:
                    # Check if v is a neighbor of cur
                    return (adj[c][cur] >> v) & 1
                
                mask = adj[c][cur] & ~visited_mask
                # Pruning: if depth == 2, we need a path cur - a - v.
                # So we need a common neighbor a of cur and v, not visited.
                if depth == 2:
                    # Common neighbors of cur and v
                    common = adj[c][cur] & adj[c][v] & ~visited_mask
                    return common != 0
                
                # If depth == 3, we need cur - a - b - v.
                # Iterate a in N(cur), check if b in N(a) & N(v) ...
                # The generic loop below handles this.
                
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if dfs(nxt, depth - 1, visited_mask | lsb):
                        return True
                    mask -= lsb
                return False

            return dfs(u, 5, 1 << u)

        def is_valid(c, u, v):
            # Adding (u,v) to color c creates a C6 if there is a path of length 5 from u to v
            return not has_path_len_5(c, u, v)

        def solve(idx):
            if time.time() > local_deadline:
                return False
            if idx == num_edges:
                return True
                
            u, v = edges[idx]
            
            # Symmetry breaking for the first few edges
            # This is crucial to reduce the search space.
            # We fix the colors of edges incident to vertex 0 and 1 to break symmetries.
            
            # Standard symmetry breaking for K_n 4-coloring:
            # 1. (0,1) is 0
            # 2. (0,2) is in {0, 1}
            # 3. (0,3) is in {0, 1, 2}
            # 4. (1,2) is in {0, 1, 2} (with constraints based on previous)
            
            # We will apply these constraints only for the first few edges.
            # The edges are generated in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
            
            if idx == 0: # (0,1)
                c = 0
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    if solve(idx + 1):
                        return True
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                return False
            
            if idx == 1: # (0,2)
                for c in [0, 1]:
                    if is_valid(c, u, v):
                        colors[idx] = c
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                        if solve(idx + 1):
                            return True
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                return False
            
            if idx == 2: # (0,3)
                for c in [0, 1, 2]:
                    if is_valid(c, u, v):
                        colors[idx] = c
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                        if solve(idx + 1):
                            return True
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                return False
            
            if idx == 3: # (0,4)
                # We can allow 0,1,2,3 here, but maybe restrict?
                # Let's allow all 4 to be safe, or restrict to 0,1,2?
                # If we restrict too much, we might miss solutions.
                # (0,4) can be any color.
                for c in range(4):
                    if is_valid(c, u, v):
                        colors[idx] = c
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                        if solve(idx + 1):
                            return True
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                return False
            
            # For (1,2), which is idx = n-1 (since (0,1)..(0,n-1) are first n-1 edges)
            # idx for (1,2) is (n-1)
            if idx == n - 1: # (1,2)
                # Colors for (1,2) can be restricted based on (0,1), (0,2), (1,3)?
                # Let's just try all 4, but maybe order them?
                for c in range(4):
                    if is_valid(c, u, v):
                        colors[idx] = c
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                        if solve(idx + 1):
                            return True
                        colors[idx] = -1
                        adj[c][u] &= ~(1 << v)
                        adj[c][v] &= ~(1 << u)
                return False

            # General case
            for c in range(4):
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if solve(idx + 1):
                        return True
                        
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            return False

        if solve(0):
            return colors
        return None

    # Try n=17 first with a generous budget
    res = attempt(17, 100)
    if res:
        return {'n': 17, 'colors': res}

    # If 17 fails, try 16
    if time.time() < hard_deadline - 10:
        res = attempt(16, 10)
        if res:
            return {'n': 16, 'colors': res}

    # Fallback to smaller n
    for n in [15, 14, 13, 12, 11, 10, 9]:
        if time.time() > hard_deadline - 2:
            break
        res = attempt(n, 2)
        if res:
            return {'n': n, 'colors': res}

    # Absolute fallback: return a small valid coloring
    # n=4 is always valid (no cycles of length 6 possible)
    n = 4
    colors = [0, 1, 2, 3, 0, 1, 2] # (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=0, (2,3)=1
    # Let's just return n=1 with empty colors? No, n>=1.
    # Let's return n=4 with a valid coloring.
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Colors: 0, 1, 2, 3, 0, 1
    # Check C6: No C6 in K4.
    return {'n': 4, 'colors': [0, 1, 2, 3, 0, 1]}
