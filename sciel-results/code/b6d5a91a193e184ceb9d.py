import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Seed solution
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    best_n = n_seed
    best_colors = colors_seed
    
    # Try to find larger solutions
    for n in range(10, 18):
        if time.time() > deadline:
            break
            
        found = False
        restarts = 0
        max_restarts = 10000
        
        while not found and restarts < max_restarts:
            if time.time() > deadline:
                break
            restarts += 1
            
            colors = [-1] * (n * (n - 1) // 2)
            adj = [[set() for _ in range(n)] for _ in range(4)]
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            # Symmetry breaking: (0,1) is color 0
            idx_01 = edges.index((0, 1))
            
            def creates_c6(u, v, c):
                for x in adj[c][u]:
                    if x == v: continue
                    for y in adj[c][x]:
                        if y == u or y == x or y == v: continue
                        for z in adj[c][y]:
                            if z == x or z == y or z == v: continue
                            if v in adj[c][z]:
                                return True
                return False

            def solve(k):
                if time.time() > deadline:
                    return False
                
                if k == len(edges):
                    return True
                
                u, v = edges[k]
                
                if k == idx_01:
                    if creates_c6(u, v, 0):
                        return False
                    colors[k] = 0
                    adj[0][u].add(v)
                    adj[0][v].add(u)
                    res = solve(k + 1)
                    if res:
                        return True
                    adj[0][u].remove(v)
                    adj[0][v].remove(u)
                    colors[k] = -1
                    return False
                
                # Heuristic: Try colors that appear less? Or random.
                # Let's try random order for diversity
                color_order = [0, 1, 2, 3]
                if restarts > 5:
                    random.shuffle(color_order)
                
                for c in color_order:
                    if creates_c6(u, v, c):
                        continue
                    
                    colors[k] = c
                    adj[c][u].add(v)
                    adj[c][v].add(u)
                    
                    if solve(k + 1):
                        return True
                    
                    adj[c][u].remove(v)
                    adj[c][v].remove(u)
                    colors[k] = -1
                    
                return False
            
            if solve(0):
                found = True
                best_n = n
                best_colors = colors[:]
        
    return {'n': best_n, 'colors': best_colors}
