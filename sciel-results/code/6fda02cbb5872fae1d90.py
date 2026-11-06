"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    n_colors = 20
    
    def count_violations(adj):
        violations = 0
        for u in range(n):
            for w in range(u + 1, n):
                common = [v for v in range(n) if v != u and v != w and adj[u][v] >= 0 and adj[w][v] >= 0]
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        v1, v2 = common[i], common[j]
                        c1, c2 = adj[u][v1], adj[v1][w]
                        c3, c4 = adj[w][v2], adj[v2][u]
                        if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                            violations += 1
        return violations
    
    def count_edges(adj):
        return sum(1 for u in range(n) for v in range(u+1, n) if adj[u][v] >= 0)
    
    def adj_to_edge_list(adj):
        return [[u, v, adj[u][v]] for u in range(n) for v in range(u+1, n) if adj[u][v] >= 0]
    
    best_adj = None
    best_edge_count = 0
    random.seed(123)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        adj = [[-1] * n for _ in range(n)]
        density = 0.3 + 0.4 * (restart % 5) / 4
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    c = random.randint(0, n_colors - 1)
                    adj[u][v] = c
                    adj[v][u] = c
        
        cur_edges = count_edges(adj)
        cur_viols = count_violations(adj)
        
        if cur_viols == 0 and cur_edges > best_edge_count:
            best_edge_count = cur_edges
            best_adj = [row[:] for row in adj]
        
        for step in range(3000):
            if time.time() > deadline:
                break
            
            added = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if adj[u][v] >= 0:
                        continue
                    for c in range(n_colors):
                        adj[u][v] = c
                        adj[v][u] = c
                        nv = count_violations(adj)
                        if nv < cur_viols:
                            cur_viols = nv
                            cur_edges += 1
                            added = True
                            if cur_viols == 0 and cur_edges > best_edge_count:
                                best_edge_count = cur_edges
                                best_adj = [row[:] for row in adj]
                            break
                        else:
                            adj[u][v] = -1
                            adj[v][u] = -1
                    if added:
                        break
                if added:
                    break
            
            if added:
                continue
            
            recolored = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if adj[u][v] < 0:
                        continue
                    old_c = adj[u][v]
                    for c in range(n_colors):
                        if c == old_c:
                            continue
                        adj[u][v] = c
                        adj[v][u] = c
                        nv = count_violations(adj)
                        if nv < cur_viols:
                            cur_viols = nv
                            recolored = True
                            break
                        else:
                            adj[u][v] = old_c
                            adj[v][u] = old_c
                    if recolored:
                        break
                if recolored:
                    break
            
            if recolored:
                continue
            
            removed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if adj[u][v] < 0:
                        continue
                    saved_c = adj[u][v]
                    adj[u][v] = -1
                    adj[v][u] = -1
                    nv = count_violations(adj)
                    if nv < cur_viols:
                        cur_viols = nv
                        cur_edges -= 1
                        removed = True
                    else:
                        adj[u][v] = saved_c
                        adj[v][u] = saved_c
                    if removed:
                        break
                if removed:
                    break
        
        if cur_viols == 0 and cur_edges > best_edge_count:
            best_edge_count = cur_edges
            best_adj = [row[:] for row in adj]
    
    if best_adj is not None:
        return adj_to_edge_list(best_adj)
    return [[0, v, 0] for v in range(1, 20)]
# EVOLVE-BLOCK-END
