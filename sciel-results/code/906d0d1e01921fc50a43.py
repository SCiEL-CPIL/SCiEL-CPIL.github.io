import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def has_path_len_5(adj_bitmasks, u, v, n):
        """Check if there's a simple path of length 5 from u to v in the graph defined by adj_bitmasks."""
        # Use BFS with state (node, depth, visited_mask)
        # Since n <= 20, we can use bitmask for visited
        # But depth is only 5, so we can use DFS
        from collections import deque
        
        # State: (node, depth, visited_mask)
        # We want to reach v at depth 5
        # Use BFS to find shortest path, but we need exactly length 5
        # Actually, we need any simple path of length 5
        # DFS with backtracking
        
        visited_mask = 1 << u
        
        def dfs(node, depth):
            if depth == 5:
                return node == v
            if depth > 5:
                return False
            # Get neighbors
            adj_mask = adj_bitmasks[node]
            # Iterate through neighbors
            while adj_mask:
                low_bit = adj_mask & (-adj_mask)
                nei = low_bit.bit_length() - 1
                adj_mask ^= low_bit
                if not (visited_mask & (1 << nei)):
                    visited_mask[0] = visited_mask[0] | (1 << nei)
                    if dfs(nei, depth + 1):
                        return True
                    visited_mask[0] = visited_mask[0] & ~(1 << nei)
            return False
        
        # Use a mutable visited_mask for the DFS
        visited_mask = [1 << u]
        return dfs(u, 0)

    def has_path_len_5_fast(adj_bitmasks, u, v, n):
        """Faster check using iterative deepening DFS with bitmask visited."""
        if u == v:
            return False
        
        # Try to find a path u = v0, v1, v2, v3, v4, v5 = v where all vertices are distinct
        # and (vi, vi+1) is an edge in the color graph
        
        # We can use a recursive DFS
        sys.setrecursionlimit(10000)
        
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            if depth >= 5:
                return False
            
            adj_mask = adj_bitmasks[node]
            # Filter out visited neighbors
            adj_mask &= ~visited
            
            # Heuristic: try neighbors that are closer to v? 
            # For now, just iterate
            while adj_mask:
                low_bit = adj_mask & (-adj_mask)
                nei = low_bit.bit_length() - 1
                adj_mask ^= low_bit
                
                if dfs(nei, depth + 1, visited | (1 << nei)):
                    return True
            
            return False
        
        return dfs(u, 0, 1 << u)

    def solve_backtrack(n, max_time=100.0):
        """Backtracking search for a valid 4-coloring of K_n."""
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [-1] * m
        
        # adj_bitmasks[color][vertex] = bitmask of neighbors of vertex in color graph
        adj_bitmasks = [[0] * n for _ in range(4)]
        
        def check_valid(c, u, v):
            """Check if adding edge (u,v) with color c creates a C6."""
            return has_path_len_5_fast(adj_bitmasks[c], u, v, n)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == m:
                return True
            
            u, v = edges[idx]
            
            for c in range(4):
                if check_valid(c, u, v):
                    continue
                
                # Assign
                colors[idx] = c
                adj_bitmasks[c][u] |= (1 << v)
                adj_bitmasks[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                colors[idx] = -1
                adj_bitmasks[c][u] &= ~(1 << v)
                adj_bitmasks[c][v] &= ~(1 << u)
            
            return False
        
        return backtrack(0), colors
    
    def solve_with_randomization(n, num_restarts=10, max_time_per_restart=10.0):
        """Try multiple restarts with randomized color orders."""
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            colors = [-1] * m
            adj_bitmasks = [[0] * n for _ in range(4)]
            
            # Shuffle color order for this restart
            color_order = list(range(4))
            random.shuffle(color_order)
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                
                if idx == m:
                    return True
                
                u, v = edges[idx]
                
                for c in color_order:
                    if has_path_len_5_fast(adj_bitmasks[c], u, v, n):
                        continue
                    
                    colors[idx] = c
                    adj_bitmasks[c][u] |= (1 << v)
                    adj_bitmasks[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    colors[idx] = -1
                    adj_bitmasks[c][u] &= ~(1 << v)
                    adj_bitmasks[c][v] &= ~(1 << u)
                
                return False
            
            if backtrack(0):
                return colors
        
        return None
    
    # Try to find the largest n
    # Start from 17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try backtracking with randomization
        result = solve_with_randomization(n, num_restarts=5, max_time_per_restart=8.0)
        
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Fallback: try smaller n with more careful search
    for n in range(12, 8, -1):
        if time.time() > deadline - 2.0:
            break
        
        result = solve_with_randomization(n, num_restarts=20, max_time_per_restart=3.0)
        
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Final fallback: n=9 with a known valid coloring
    # For n=9, we can use a simple cyclic coloring
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use a pattern that avoids C6
            # The cyclic coloring (j - i) % 4 might work for small n
            # Let's verify: for n=9, K9 with colors (j-i)%4
            # Does any color class contain C6?
            # Color c edges: (i,j) where j-i ≡ c (mod 4)
            # For c=0: edges (i, i+4) for i=0..4, and (i, i+8) for i=0..0
            #   Vertices: 0-4, 1-5, 2-6, 3-7, 4-8, 0-8
            #   This is a path 8-0-4 and 8-4? No, 0-4, 4-8, 0-8 forms triangle 0-4-8
            #   Also 1-5, 2-6, 3-7 are isolated edges
            #   No C6 in this color class
            # Similar analysis for other colors should be fine for n=9
            colors.append((j - i) % 4)
    
    # Verify the fallback
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    # Check no C6
    valid = True
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if has_path_len_5_fast(adj[c], i, j, n):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {'n': n, 'colors': colors}
    
    # If even n=9 fallback fails (shouldn't happen), return n=1
    return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res)
