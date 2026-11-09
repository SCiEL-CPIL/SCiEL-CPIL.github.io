import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    deadline = time.time() + 115
    random.seed(42)
    np.random.seed(42)
    
    best_count = 0
    best_adj = None
    
    # Precompute all pairs
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_forbidden_vectorized(u, v, adj):
        """
        Returns a boolean array of forbidden colors for edge (u, v).
        Uses vectorized operations to check all 4-cycles.
        """
        # Find all x where edge (v,x) exists
        valid_x = np.where((adj[v, :] != -1) & (np.arange(n) != u) & (np.arange(n) != v))[0]
        
        if len(valid_x) == 0:
            return np.zeros(n, dtype=bool)
        
        forbidden = np.zeros(n, dtype=bool)
        
        for x in valid_x:
            # Find all y where edges (x,y) and (y,u) exist
            valid_y = np.where((adj[x, :] != -1) & (adj[:, u] != -1) & 
                             (np.arange(n) != u) & (np.arange(n) != v) & 
                             (np.arange(n) != x))[0]
            
            if len(valid_y) == 0:
                continue
            
            for y in valid_y:
                c1 = adj[v, x]
                c2 = adj[x, y]
                c3 = adj[y, u]
                
                # If any of the three existing colors are the same, 
                # then NO color works for (u,v)
                if c1 == c2 or c1 == c3 or c2 == c3:
                    return np.ones(n, dtype=bool)
                
                forbidden[c1] = True
                forbidden[c2] = True
                forbidden[c3] = True
        
        return forbidden

    def is_recolor_valid_vectorized(u, v, new_c, adj):
        """
        Check if recoloring existing edge (u, v) to new_c is valid.
        """
        valid_x = np.where((adj[v, :] != -1) & (np.arange(n) != u) & (np.arange(n) != v))[0]
        
        for x in valid_x:
            valid_y = np.where((adj[x, :] != -1) & (adj[:, u] != -1) & 
                             (np.arange(n) != u) & (np.arange(n) != v) & 
                             (np.arange(n) != x))[0]
            
            for y in valid_y:
                c1 = adj[v, x]
                c2 = adj[x, y]
                c3 = adj[y, u]
                
                if new_c == c1 or new_c == c2 or new_c == c3:
                    return False
        return True

    def check_all_4cycles_valid(adj):
        """
        Verify that all 4-cycles in the graph are rainbow.
        Returns True if valid, False otherwise.
        """
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == -1:
                    continue
                for x in range(n):
                    if x == i or x == j:
                        continue
                    if adj[j, x] == -1:
                        continue
                    for y in range(n):
                        if y == i or y == j or y == x:
                            continue
                        if adj[x, y] == -1 or adj[y, i] == -1:
                            continue
                        
                        c1 = adj[i, j]
                        c2 = adj[j, x]
                        c3 = adj[x, y]
                        c4 = adj[y, i]
                        
                        if len({c1, c2, c3, c4}) < 4:
                            return False
        return True

    for restart in range(50):
        if time.time() > deadline:
            break
            
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Greedy Phase
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            forb = get_forbidden_vectorized(u, v, adj)
            if not forb.all():
                available = np.where(~forb)[0]
                c = random.choice(available.tolist())
                adj[u, v] = c
                adj[v, u] = c
        
        # Local Search Phase with vectorized checks
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                if adj[u, v] != -1:
                    continue
                
                forb = get_forbidden_vectorized(u, v, adj)
                if not forb.all():
                    available = np.where(~forb)[0]
                    c = random.choice(available.tolist())
                    adj[u, v] = c
                    adj[v, u] = c
                    improved = True
            
            if not improved:
                # Try recoloring with vectorized check
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
                random.shuffle(existing)
                
                for u, v in existing[:10]:
                    if time.time() > deadline:
                        break
                    
                    old_c = adj[u, v]
                    colors = list(range(n))
                    random.shuffle(colors)
                    
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        
                        if is_recolor_valid_vectorized(u, v, new_c, adj):
                            adj[u, v] = new_c
                            adj[v, u] = new_c
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if not improved:
                # Stuck in this restart
                break
        
        cnt = count_edges(adj)
        
        # Verify validity before accepting
        if check_all_4cycles_valid(adj):
            if cnt > best_count:
                best_count = cnt
                best_adj = adj.copy()
        
        if best_count > 150:
            break

    # Convert to result
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
