import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def try_construct_n(n, seed, time_limit):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=bool)
        forbidden = np.zeros((n, n), dtype=bool)
        
        # List of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Set of available edges for O(1) removal
        available = set(all_edges)
        
        deg = np.zeros(n, dtype=int)
        
        while available:
            if time.time() > time_limit:
                break
            
            # Heuristic: Pick an edge that helps balance degrees
            # Sample a few candidates and pick the one with lowest sum of degrees
            # This is faster than sorting all available edges
            candidates = rng.sample(list(available), min(10, len(available)))
            best_edge = None
            min_score = float('inf')
            for u, v in candidates:
                score = deg[u] + deg[v]
                if score < min_score:
                    min_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                break
                
            u, v = best_edge
            
            # Add edge
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            
            available.discard(best_edge)
            
            # Update forbidden edges
            # New paths of length 3: x-u-v-y
            # x in N(u), y in N(v)
            nu = np.where(adj[u])[0]
            nv = np.where(adj[v])[0]
            
            for x in nu:
                if x == v:
                    continue
                for y in nv:
                    if y == u:
                        continue
                    if x == y:
                        continue
                    # Edge (x,y) is now forbidden
                    if not forbidden[x][y]:
                        forbidden[x][y] = True
                        forbidden[y][x] = True
                        # Remove from available
                        if x < y:
                            available.discard((x, y))
                        else:
                            available.discard((y, x))
                            
        # Check min degree
        min_deg = int(deg.min())
        if min_deg >= n - 39:
            # Extract edges
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append([i, j])
            return {'n': n, 'edges': edge_list}
        return None

    best_result = {'n': 39, 'edges': []}
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # More restarts for higher n as it's harder
        max_restarts = 200 if n >= 46 else 100
        
        for seed in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            res = try_construct_n(n, seed, deadline - 10)
            if res:
                best_result = res
                found = True
                break
        
        if found:
            return best_result
            
    return best_result
