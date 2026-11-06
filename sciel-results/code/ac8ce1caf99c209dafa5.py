import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def has_path_len_5(adj_c, u, v):
        """
        Check if there is a simple path of length 5 from u to v in the graph defined by adj_c.
        The path is u - x1 - x2 - x3 - x4 - v.
        adj_c is a list of bitmasks.
        """
        # We want to find x1, x2, x3, x4 such that:
        # (u, x1) in E, (x1, x2) in E, (x2, x3) in E, (x3, x4) in E, (x4, v) in E
        # All vertices distinct.
        
        # Optimization: Iterate over neighbors of u and v
        # x1 is neighbor of u
        # x4 is neighbor of v
        # Path x1 - x2 - x3 - x4 of length 3.
        
        # Let's do a DFS from u to v with depth 5.
        # State: (current_node, depth, visited_mask)
        # visited_mask is an int bitmask.
        
        # To speed up, we can prune if current_node is v and depth < 5 (invalid for simple path C6? 
        # Actually, if we reach v early, we can't continue because v is the end. 
        # But we must ensure we don't use v as an intermediate node.
        # So if current_node == v, return False (unless depth==5, but we check at end).
        
        stack = [(u, 0, 1 << u)]
        while stack:
            node, depth, visited = stack.pop()
            if depth == 5:
                if node == v:
                    return True
                continue
            
            # Get neighbors
            neighbors = adj_c[node] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                
                # If next_node is v, we can only accept it if depth+1 == 5
                # If depth+1 < 5, we cannot go through v
                if next_node == v:
                    if depth + 1 == 5:
                        return True
                    else:
                        continue
                        
                stack.append((next_node, depth + 1, visited | lsb))
        
        return False

    def solve_backtrack(n, time_limit):
        """
        Solve for K_n using backtracking.
        Returns list of colors for edges in lex order, or None.
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        # Symmetry breaking:
        # Fix colors of edges (0,1), (0,2), (0,3), (0,4) to 0, 1, 2, 3
        # This assumes n >= 5.
        fixed_edges = {}
        if n >= 5:
            fixed_edges[(0, 1)] = 0
            fixed_edges[(0, 2)] = 1
            fixed_edges[(0, 3)] = 2
            fixed_edges[(0, 4)] = 3
            
            # Initialize adj for fixed edges
            for (i, j), c in fixed_edges.items():
                idx = edges.index((i, j))
                colors[idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Find start index
        start_idx = 0
        while start_idx < num_edges and colors[start_idx] != -1:
            start_idx += 1
            
        if start_idx == num_edges:
            return colors
            
        # Recursive backtrack
        # To avoid recursion limit issues and overhead, we might use an iterative approach 
        # or just set recursion limit high.
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            if time.time() > time_limit:
                return None
                
            if idx == num_edges:
                return colors[:]
                
            # Skip fixed edges
            if colors[idx] != -1:
                return backtrack(idx + 1)
                
            u, v = edges[idx]
            
            # Try colors 0, 1, 2, 3
            for c in range(4):
                # Check if adding (u,v) to color c creates a C6
                # We need to check if there is a path of length 5 from u to v in adj[c]
                # Note: adj[c] currently does NOT contain (u,v)
                if has_path_len_5(adj[c], u, v):
                    continue
                    
                # Assign
                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                    
                # Undo
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = -1
                
            return None
            
        return backtrack(start_idx)

    def solve_local_search(n, time_limit):
        """
        Solve for K_n using local search (Simulated Annealing / Random Restart).
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        def count_c6(colors):
            """Count total monochromatic C6s."""
            adj = [[0] * n for _ in range(4)]
            for k, (i, j) in enumerate(edges):
                c = colors[k]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
            count = 0
            # For each color, check for C6
            # Efficient check: For each edge (u,v) in color c, check if path len 5 exists?
            # No, that counts each C6 6 times (once per edge).
            # We just need to know if count is 0.
            # We can stop early if we find one? No, we want to minimize.
            # But for SA, we need the cost.
            # Cost = number of C6s.
            
            # Alternative: Precompute all C6s in K_n?
            # Too many.
            
            # Let's just use the has_path_len_5 check on the *current* graph.
            # If we add an edge, we check if it creates a C6.
            # In local search, we flip an edge.
            # Flipping (u,v) from c1 to c2:
            # 1. Remove (u,v) from c1. This might remove C6s that used (u,v).
            # 2. Add (u,v) to c2. This might create C6s that use (u,v).
            
            # This incremental update is complex to implement correctly for counting.
            # Given time limits, a full recount might be too slow for n=17.
            # \binom{17}{6} * 4 * (check cost).
            # Check cost for one C6 set: 60 permutations? No, just check if Hamiltonian cycle exists.
            # That's expensive.
            
            # Let's stick to Backtracking which is exact and prunes early.
            # Local search might be too slow to evaluate the cost function.
            return 0 # Placeholder

    # Strategy:
    # 1. Try Backtracking for n=17 with 20s limit.
    # 2. If fails, try n=16 with 20s limit.
    # ...
    # 3. If all fail, return best found or fallback.

    best_result = None
    best_n = 0
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on n
        # Larger n gets more time? Or fixed?
        # Let's give 15s for n=17, 10s for 16, etc.
        time_budget = 15.0 if n == 17 else 10.0
        
        res = solve_backtrack(n, time.time() + time_budget)
        
        if res is not None:
            # Verify
            # (Omitted for brevity, assuming backtrack is correct)
            return {'n': n, 'colors': res}
            
    # Fallback to a known small solution
    # n=9 pattern from seed
    n_fb = 9
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    return {'n': n_fb, 'colors': colors_fb}
