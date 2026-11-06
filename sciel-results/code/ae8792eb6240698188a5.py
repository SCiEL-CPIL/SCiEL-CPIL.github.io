import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer

    def has_simple_path_len_5(adj, c, u, v):
        # adj[c] is a list of bitmasks
        # Check if there is a simple path of length 5 from u to v in graph c
        
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            if node == v:
                return False # Path too short
            
            neighbors = adj[c][node] & ~visited
            
            # Pruning: if not enough neighbors to complete the path
            # We need (5 - depth) more steps
            # Each step requires at least 1 neighbor
            # So we need at least (5 - depth) neighbors
            if neighbors.bit_count() < (5 - depth):
                return False
                
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    def solve(n, time_limit):
        # adj[c][u] is bitmask of neighbors
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        # Symmetry breaking
        fixed_edges = [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (1, 2, 0),
            (1, 3, 1),
            (2, 3, 1)
        ]
        
        # Apply fixed edges
        for u, v, c in fixed_edges:
            if u >= n or v >= n:
                continue
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[u][v] = c
            colors[v][u] = c
            
        # Determine the next edge to color
        # We iterate through all edges (i, j) with i < j
        # and skip the fixed ones.
        
        edges_to_color = []
        for i in range(n):
            for j in range(i + 1, n):
                if colors[i][j] == -1:
                    edges_to_color.append((i, j))
                    
        def backtrack(idx):
            if time.time() > time_limit:
                return None
            
            if idx == len(edges_to_color):
                return colors # All edges colored
            
            u, v = edges_to_color[idx]
            
            # Try all 4 colors for (i, j), randomized
            cs = [0, 1, 2, 3]
            random.shuffle(cs)
            
            for c in cs:
                if has_simple_path_len_5(adj, c, u, v):
                    continue
                
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                colors[u][v] = c
                colors[v][u] = c
                
                res = backtrack(idx + 1)
                if res:
                    return res
                
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[u][v] = -1
                colors[v][u] = -1
                
            return None
        
        return backtrack(0)

    best_result = None
    
    # Try n=18 with a short time budget
    if time.time() < deadline - 10:
        t_limit = time.time() + 5
        res = solve(18, t_limit)
        if res:
            flat_colors = []
            for i in range(18):
                for j in range(i + 1, 18):
                    flat_colors.append(res[i][j])
            best_result = {'n': 18, 'colors': flat_colors}
            
    # Try n=17 with multiple restarts
    if not best_result:
        t_limit_total = deadline - 10
        while time.time() < t_limit_total:
            t_limit = time.time() + 10 # 10s per restart
            if t_limit > t_limit_total:
                t_limit = t_limit_total
            res = solve(17, t_limit)
            if res:
                flat_colors = []
                for i in range(17):
                    for j in range(i + 1, 17):
                        flat_colors.append(res[i][j])
                best_result = {'n': 17, 'colors': flat_colors}
                break
                
    # Try n=16
    if not best_result:
        t_limit = deadline - 2
        res = solve(16, t_limit)
        if res:
            flat_colors = []
            for i in range(16):
                for j in range(i + 1, 16):
                    flat_colors.append(res[i][j])
            best_result = {'n': 16, 'colors': flat_colors}
            
    if best_result:
        return best_result
        
    # Fallback: Return a known valid small coloring
    # n=9 is valid in the seed
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
