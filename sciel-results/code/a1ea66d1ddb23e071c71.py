import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Seed solution
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    best_n = n_seed
    best_colors = colors_seed
    
    # Precompute edge data for n=10..17
    edge_data = {}
    for n in range(10, 18):
        edges = []
        idx_map = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
                idx_map[i][j] = idx
                idx_map[j][i] = idx
                idx += 1
        edge_data[n] = (edges, idx_map)
        
    def count_paths_4(u, v, A, n):
        cnt = 0
        # A is a list of lists
        for w1 in range(n):
            if w1 == u or w1 == v: continue
            if A[u][w1] == 0: continue
            for w2 in range(n):
                if w2 == u or w2 == v or w2 == w1: continue
                if A[w1][w2] == 0: continue
                for w3 in range(n):
                    if w3 == u or w3 == v or w3 == w1 or w3 == w2: continue
                    if A[w2][w3] == 0: continue
                    if A[w3][v] == 1:
                        cnt += 1
        return cnt

    def count_c6_in_color(A, n):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if A[u][v] == 1:
                    count += count_paths_4(u, v, A, n)
        return count // 6

    def has_c6_exact(colors, n):
        adj = [[0]*n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i][j] = 1
                adj[c][j][i] = 1
                idx += 1
        for c in range(4):
            if count_c6_in_color(adj[c], n) > 0:
                return True
        return False

    # Search for higher n
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        edges, idx_map = edge_data[n]
        m = len(edges)
        
        # Time allocation
        time_alloc = min(20, remaining * 0.5)
        
        # Try to find a solution
        found = False
        
        # Multiple restarts
        max_restarts = 5
        restart_time = time_alloc / max_restarts
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            colors = [random.randint(0, 3) for _ in range(m)]
            adj = [[0]*n for _ in range(4)]
            
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj[c][i][j] = 1
                adj[c][j][i] = 1
                
            cost = 0
            for c in range(4):
                cost += count_c6_in_color(adj[c], n)
                
            if cost == 0:
                found = True
                break
                
            end_time = time.time() + restart_time
            while time.time() < end_time:
                idx = random.randint(0, m - 1)
                i, j = edges[idx]
                old_c = colors[idx]
                
                # Pick new color
                new_c = random.choice([0, 1, 2, 3])
                if new_c == old_c:
                    continue
                    
                k_old = count_paths_4(i, j, adj[old_c], n)
                k_new = count_paths_4(i, j, adj[new_c], n)
                
                delta = k_new - k_old
                
                if delta < 0:
                    adj[old_c][i][j] = 0
                    adj[old_c][j][i] = 0
                    adj[new_c][i][j] = 1
                    adj[new_c][j][i] = 1
                    colors[idx] = new_c
                    cost += delta
                    if cost == 0:
                        found = True
                        break
                        
        if found:
            if not has_c6_exact(colors, n):
                best_n = n
                best_colors = colors
                break # Found highest n, stop
            
    return {'n': best_n, 'colors': best_colors}
