import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def has_path_len_5(adj_c, u, v):
        def dfs(current, depth, visited):
            if depth == 5:
                return current == v
            if current == v:
                return False
            
            neighbors = adj_c[current] & ~visited
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                if dfs(next_node, depth + 1, visited | lsb):
                    return True
            return False
        
        return dfs(u, 0, 1 << u)

    def greedy_solve(n):
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                found = False
                color_order = [0, 1, 2, 3]
                random.shuffle(color_order)
                for c in color_order:
                    if has_path_len_5(adj[c], i, j):
                        continue
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    colors[i][j] = c
                    colors[j][i] = c
                    found = True
                    break
                if not found:
                    return None
        return colors

    def solve(n):
        adj = [[0] * n for _ in range(4)]
        colors = [[-1] * n for _ in range(n)]
        
        def backtrack(i, j):
            if time.time() > deadline:
                return None
            
            if i == n - 1:
                if j == n:
                    return colors
            
            if j == n:
                res = backtrack(i + 1, i + 2)
                if res: return res
                return None
            
            if i == 0 and j == 1:
                color_range = [0]
            elif i == 0 and j == 2:
                color_range = [0, 1]
            else:
                color_range = [0, 1, 2, 3]
                
            for c in color_range:
                if has_path_len_5(adj[c], i, j):
                    continue
                
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[i][j] = c
                colors[j][i] = c
                
                res = backtrack(i, j + 1)
                if res:
                    return res
                
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[i][j] = -1
                colors[j][i] = -1
            
            return None
        
        return backtrack(0, 1)

    # Try Greedy for high n first
    for n in [16, 15, 14, 13, 12]:
        if time.time() > deadline - 10:
            break
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            res = greedy_solve(n)
            if res:
                flat_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        flat_colors.append(res[i][j])
                return {'n': n, 'colors': flat_colors}
        
    # If Greedy fails, try Backtracking for smaller n
    for n in [12, 11, 10]:
        if time.time() > deadline - 5:
            break
        res = solve(n)
        if res:
            flat_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    flat_colors.append(res[i][j])
            return {'n': n, 'colors': flat_colors}
            
    # Fallback
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
