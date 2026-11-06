import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    def _has_path_len_5(adj, u, v):
        """Check if there's a simple path of length 5 from u to v using only edges in adj."""
        def dfs(current, depth, visited):
            if depth == 5:
                return current == v
            if current == v:
                return False
            
            neighbors = adj[current] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)
    
    def backtrack_solve(n, max_time=None):
        """Backtracking solver for n vertices."""
        if max_time is None:
            max_time = deadline
        
        colors = [[-1] * n for _ in range(n)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # For efficiency, maintain per-color adjacency as bitmasks
        adj = [[0] * n for _ in range(4)]
        
        def would_create_c6(c, u, v):
            """Check if adding edge (u,v) with color c creates a monochromatic C6."""
            # Check if there's a path of length 5 from u to v in current color-c graph
            def dfs(current, depth, visited):
                if depth == 5:
                    return current == v
                if current == v:
                    return False
                
                neighbors = adj[c][current] & ~visited
                while neighbors:
                    lsb = neighbors & -neighbors
                    next_node = lsb.bit_length() - 1
                    neighbors ^= lsb
                    if dfs(next_node, depth + 1, visited | lsb):
                        return True
                return False
            
            return dfs(u, 0, 1 << u)
        
        backtrack_count = [0]
        
        def backtrack(idx):
            if time.time() > max_time:
                return None
            
            if idx == len(edges):
                return colors
            
            backtrack_count[0] += 1
            
            i, j = edges[idx]
            
            # Try colors 0,1,2,3
            for c in range(4):
                if would_create_c6(c, i, j):
                    continue
                
                # Assign color c to (i,j)
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                res = backtrack(idx + 1)
                if res:
                    return res
                
                # Undo
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
            
            return None
        
        return backtrack(0)
    
    # Try to find a valid coloring, starting from n=17 down
    for n in [17, 16, 15]:
        if time.time() > deadline - 5:
            break
        
        # Give each n a time budget
        time_budget = min(30, deadline - time.time() - 5)
        if time_budget <= 0:
            break
        
        max_time = time.time() + time_budget
        result = backtrack_solve(n, max_time=max_time)
        if result is not None:
            flat_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat_colors.append(result[i][j])
            return {'n': n, 'colors': flat_colors}
    
    # Fallback: return a small valid coloring
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}
