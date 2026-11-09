# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 20
    max_colors = 20
    full_mask = (1 << max_colors) - 1
    
    def solve_once(seed):
        random.seed(seed)
        adj = [[-1]*n for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            forbidden_mask = 0
            
            # Check for 4-cycles u-v-w-x-u
            # Edges: (u,v) [new], (v,w), (w,x), (x,u) [existing]
            for w in range(n):
                if w == u or w == v:
                    continue
                c_vw = adj[v][w]
                if c_vw == -1:
                    continue
                
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    c_wx = adj[w][x]
                    if c_wx == -1:
                        continue
                    
                    c_xu = adj[x][u]
                    if c_xu == -1:
                        continue
                    
                    # Found a cycle. Colors c_vw, c_wx, c_xu are forbidden for c
                    forbidden_mask |= (1 << c_vw)
                    forbidden_mask |= (1 << c_wx)
                    forbidden_mask |= (1 << c_xu)
                    
                    if forbidden_mask == full_mask:
                        break
                if forbidden_mask == full_mask:
                    break
            
            available_mask = full_mask & ~forbidden_mask
            if available_mask == 0:
                continue
            
            # Select a random valid color
            # Using list comprehension for clarity and speed with small N
            avail_colors = [c for c in range(max_colors) if (available_mask >> c) & 1]
            c = random.choice(avail_colors)
            
            adj[u][v] = c
            adj[v][u] = c
            
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] != -1:
                    result.append([i, j, adj[i][j]])
        return result

    best_edges = []
    start_time = time.time()
    
    # Run multiple restarts
    # Estimate time per restart: ~1-5ms?
    # 120s allows for many restarts.
    for i in range(1000):
        if time.time() - start_time > 110:
            break
        res = solve_once(i)
        if len(res) > len(best_edges):
            best_edges = res
            
    return best_edges
# EVOLVE-BLOCK-END
