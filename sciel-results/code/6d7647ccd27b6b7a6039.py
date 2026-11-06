import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Seed solution for n=9
    def get_seed_colors(n):
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    best_n = 9
    best_colors = get_seed_colors(9)
    
    def edge_index(i, j, n):
        # Index in the list of colors for pair (i, j) with i < j
        # Total edges before row i: i*(n-1) - i*(i-1)/2 ?
        # Let's just use a dictionary or a precomputed list for speed if n is small, 
        # but a formula is better.
        # Number of edges (k, l) with k < i:
        # For k=0..i-1, number of l > k is n-1-k.
        # Sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # Edges in row i before j: j - i - 1
        # Total index = i*(n-1) - i*(i-1)//2 + (j - i - 1)
        return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)

    def build_adj_lists(colors, n):
        adj = [[[] for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_index(i, j, n)
                c = colors[idx]
                adj[c][i].append(j)
                adj[c][j].append(i)
        return adj

    def find_c6(adj, color):
        # Returns a list of 6 vertices forming a C6 in 'color' graph, or None
        graph = adj[color]
        n = len(graph)
        
        # Sort neighbors for determinism? Not needed, just speed.
        for start in range(n):
            # DFS to find path of length 5 from start back to start
            # Path: start -> v1 -> v2 -> v3 -> v4 -> v5 -> start
            # We track visited nodes to ensure simple cycle
            visited = [False] * n
            path = []
            
            def dfs(node, depth):
                if depth == 5:
                    # Check if node is connected to start
                    if start in graph[node]:
                        return path + [start]
                    return None
                
                for neighbor in graph[node]:
                    if neighbor == start:
                        # Only allowed if depth == 5 (handled above) 
                        # If depth < 5, we can't close the cycle yet
                        # But wait, if depth < 5 and neighbor is start, it's a shorter cycle. 
                        # We want exactly 6. So we must NOT visit start until the end.
                        continue
                    
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        path.append(neighbor)
                        res = dfs(neighbor, depth + 1)
                        if res:
                            return res
                        path.pop()
                        visited[neighbor] = False
                return None
            
            visited[start] = True
            res = dfs(start, 0)
            if res:
                return res
            visited[start] = False
            
        return None

    def has_c6(colors, n):
        adj = build_adj_lists(colors, n)
        for c in range(4):
            cycle = find_c6(adj, c)
            if cycle:
                return cycle
        return None

    def try_find_coloring(n, time_limit):
        # Local search with restarts
        num_edges = n * (n - 1) // 2
        while time.time() < time_limit:
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Initial check
            cycle = has_c6(colors, n)
            if not cycle:
                return colors
            
            # Local improvement
            stuck_count = 0
            while time.time() < time_limit and stuck_count < 1000:
                cycle = has_c6(colors, n)
                if not cycle:
                    return colors
                
                # Pick a random edge from the cycle and change its color
                # Cycle is [v0, v1, v2, v3, v4, v5, v0]
                # Edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
                edge_idx_in_cycle = random.randint(0, 5)
                u = cycle[edge_idx_in_cycle]
                v = cycle[(edge_idx_in_cycle + 1) % 6]
                
                if u > v:
                    u, v = v, u
                    
                idx = edge_index(u, v, n)
                old_color = colors[idx]
                new_color = random.randint(0, 3)
                while new_color == old_color:
                    new_color = random.randint(0, 3)
                
                colors[idx] = new_color
                
                # If we didn't fix the specific cycle, we might have created a new one or kept it.
                # To avoid infinite loops on the same cycle, we count "stuck" if we revert?
                # Simple heuristic: just keep going.
                stuck_count += 1
                
            # If loop finished without return, we failed this restart
            pass
        
        return None

    # Search for increasing n
    # We allocate time based on remaining budget
    # Start with n=10
    for n in range(10, 21):
        # Allocate a portion of remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Heuristic: give more time to smaller n, less to larger n?
        # Or just a fixed slice, e.g., 15s per n, but capped by remaining.
        slice_time = min(15, remaining - 2) # Keep 2s safety
        if slice_time <= 0:
            break
            
        print(f"Trying n={n} with {slice_time:.1f}s budget")
        colors = try_find_coloring(n, time.time() + slice_time)
        
        if colors:
            best_n = n
            best_colors = colors
            print(f"Found valid coloring for n={n}")
        else:
            # If we fail for n, we might still try n+1? Unlikely to succeed if n failed.
            # But maybe n=10 failed due to bad luck, n=11 might succeed? No, n=11 is harder.
            # So we break.
            break
            
    return {'n': best_n, 'colors': best_colors}
