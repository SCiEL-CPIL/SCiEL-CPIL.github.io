import sys
import time

def construct():
    start_time = time.time()
    time_limit = 115.0 # Leave 5s for overhead
    
    # Try n=17
    n = 17
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Graphs for each color
    # Using list of sets for adjacency to check paths? 
    # Or just list of edges and check C6 via BFS on the fly?
    # BFS on the fly is safer and simpler.
    
    def has_c6_completion(color, u, v, graphs):
        # Check if adding (u,v) to graphs[color] creates a C6.
        # This is true if there is a path of length 5 between u and v in graphs[color].
        # BFS from u up to depth 5.
        # If we reach v at depth 5, we have a C6.
        # Note: The path must be simple. BFS guarantees simple path in unweighted graph if we track visited.
        
        # Optimization: If degree of u or v is low, it's fast.
        # Graphs[color] is a list of edges. We need adjacency.
        # We will maintain adjacency lists for speed.
        
        pass # Placeholder

    # Better: Maintain adjacency lists
    adj = [set() for _ in range(4)] # adj[c][u] is set of neighbors
    
    def is_safe(c, u, v):
        # Check if adding edge (u,v) to color c creates a C6.
        # A C6 is u - ... - v - u.
        # The path ... must be of length 5.
        # So we need a path of length 5 from u to v in the current graph of color c.
        
        # BFS from u
        # dist[v] = 5?
        # We need to check if there is a path of length exactly 5.
        # Since the graph is small, we can just do BFS and check if v is reachable at depth 5.
        # However, BFS finds shortest paths. If shortest path is < 5, there might still be a path of length 5.
        # Example: Triangle u-w-x-u. Path length 2. But if there is another path of length 5, that's a C6?
        # No, a C6 is a cycle. The edge (u,v) closes the cycle.
        # So we need a path of length 5 between u and v that does NOT use the edge (u,v) (which isn't there yet).
        # And the vertices in the path must be distinct from u and v?
        # Yes, a simple path.
        
        # So: Is there a simple path of length 5 from u to v in G_c?
        
        # Since max degree is small, we can enumerate paths of length 5?
        # Or BFS with depth limit 5.
        # BFS gives shortest distance. If dist < 5, it doesn't mean no path of length 5 exists.
        # We need to check if v is reachable in exactly 5 steps.
        # This is equivalent to checking if there is a walk of length 5? No, simple path.
        
        # Given the sparsity, we can just do a DFS up to depth 5.
        
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            for neighbor in adj[c]:
                if neighbor == u or neighbor == v:
                    continue # u and v are endpoints, cannot be in the middle
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if dfs(neighbor, depth + 1, visited):
                    return True
                visited.remove(neighbor)
            return False
        
        visited = {u}
        return dfs(u, 0, visited)

    # This DFS is called for every edge assignment.
    # Number of edges 136.
    # Depth 5.
    # It should be fast enough.
    
    # Backtracking
    colors = [-1] * len(edges)
    
    def backtrack(idx):
        if time.time() - start_time > time_limit:
            return False # Time out
        
        if idx == len(edges):
            return True
            
        u, v = edges[idx]
        
        # MRV: Try colors in order of fewest "failures" or just 0,1,2,3
        # To break symmetry, we can restrict choices for early edges.
        
        # Simple heuristic: Try 0,1,2,3
        for c in range(4):
            if is_safe(c, u, v):
                # Assign
                colors[idx] = c
                adj[c].add((u, v)) # Wait, adj is set of neighbors, not edges.
                # Let's update adj properly.
                # adj[c] is a list of sets.
                # adj[c][u].add(v)
                # adj[c][v].add(u)
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                colors[idx] = -1
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                
        return False

    # Initialize adj
    adj = [ [set() for _ in range(n)] for _ in range(4) ]
    
    # Symmetry breaking:
    # Fix edge (0,1) to color 0.
    # This reduces search space by 4.
    # We can fix more.
    
    # Let's try to run backtrack.
    # But we need to handle the time limit gracefully.
    # If backtrack returns False due to time limit, we should return a smaller n.
    
    # To be safe, let's try n=10 first.
    # If we have time, try n=11, 12... up to 17.
    
    # Actually, if we just try n=17 and it fails, we return n=9?
    # That's a waste.
    
    # Strategy:
    # 1. Try to find solution for n=17.
    # 2. If found, return.
    # 3. If not found within time_limit/2, try n=16?
    # 4. If not, try n=15?
    # ...
    # 5. If not, return n=9 (known valid).
    
    # But finding n=16 might also take long.
    # Let's just try n=17. If it works, great.
    # If not, we need a fallback that is fast.
    # We can generate a random coloring for n=10 and check if it's valid.
    # Or use the n=9 construction.
    
    # Let's implement the backtracker for a given n.
    
    def solve(n):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [ [set() for _ in range(n)] for _ in range(4) ]
        colors = [-1] * len(edges)
        
        def is_safe(c, u, v):
            # DFS for path of length 5
            def dfs(node, depth, visited):
                if depth == 5:
                    return node == v
                for neighbor in adj[c][node]:
                    if neighbor == u or neighbor == v:
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if dfs(neighbor, depth + 1, visited):
                        return True
                    visited.remove(neighbor)
                return False
            
            visited = {u}
            return dfs(u, 0, visited)

        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == len(edges):
                return True
                
            u, v = edges[idx]
            
            # Heuristic: Try colors that are "least constrained" or just 0-3
            # For first few edges, break symmetry
            if idx == 0:
                candidates = [0]
            elif idx == 1:
                # Edge (0,2). Can be 0 or 1?
                # If (0,1) is 0, (0,2) can be 0 or 1.
                # If we fix to 0, we might lose solutions where it's 1.
                # But color symmetry allows us to swap 0 and 1?
                # No, we fixed (0,1) to 0. We can still swap 1 and 2 and 3.
                # So (0,2) can be 0 or 1.
                candidates = [0, 1]
            else:
                candidates = [0, 1, 2, 3]
                
            for c in candidates:
                if is_safe(c, u, v):
                    colors[idx] = c
                    adj[c][u].add(v)
                    adj[c][v].add(u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    colors[idx] = -1
                    adj[c][u].remove(v)
                    adj[c][v].remove(u)
                    
            return False

        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    # Try n=17
    res = solve(17)
    if res:
        return res

    # If 17 fails (likely due to time or no solution found in time), try smaller
    # But we need to be careful with time.
    # Let's just return n=9 if we can't find 17 quickly.
    # Actually, we can try n=10 which is very fast.
    
    # To maximize score, we should try as high as possible.
    # But if 17 takes 100s and fails, we have 20s left.
    # We can try 16.
    
    # Let's modify solve to take a time budget.
    
    def solve_with_budget(n, budget):
        start = time.time()
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [ [set() for _ in range(n)] for _ in range(4) ]
        colors = [-1] * len(edges)
        
        def is_safe(c, u, v):
            def dfs(node, depth, visited):
                if depth == 5:
                    return node == v
                for neighbor in adj[c][node]:
                    if neighbor == u or neighbor == v:
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if dfs(neighbor, depth + 1, visited):
                        return True
                    visited.remove(neighbor)
                return False
            visited = {u}
            return dfs(u, 0, visited)

        def backtrack(idx):
            if time.time() - start > budget:
                return False
            if idx == len(edges):
                return True
            u, v = edges[idx]
            if idx == 0:
                candidates = [0]
            elif idx == 1:
                candidates = [0, 1]
            else:
                candidates = [0, 1, 2, 3]
                
            for c in candidates:
                if is_safe(c, u, v):
                    colors[idx] = c
                    adj[c][u].add(v)
                    adj[c][v].add(u)
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
                    adj[c][u].remove(v)
                    adj[c][v].remove(u)
            return False

        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    # Try 17 with 100s
    res = solve_with_budget(17, 100.0)
    if res:
        return res
        
    # Try 16 with 10s
    res = solve_with_budget(16, 10.0)
    if res:
        return res
        
    # Try 15 with 5s
    res = solve_with_budget(15, 5.0)
    if res:
        return res
        
    # Try 10 with 1s (should be instant)
    res = solve_with_budget(10, 1.0)
    if res:
        return res
        
    # Fallback to known n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
