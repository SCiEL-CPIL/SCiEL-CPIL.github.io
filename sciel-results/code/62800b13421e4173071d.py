import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    best_n = 39
    
    def try_construct(n, seed):
        random.seed(seed)
        target_deg = n - 39
        
        # Try different densities
        # We want to start with enough edges to survive removals.
        # Max edges for C4-free is ~ n^2/4.
        # Min edges for target deg is n * target_deg / 2.
        # For n=46, target=7, min edges=161.
        # Turan bound ~ 529.
        # Random graph p=0.2 has ~ 414 edges.
        # p=0.15 has ~ 310 edges.
        
        densities = [0.25, 0.22, 0.20, 0.18, 0.16, 0.14]
        
        for p in densities:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Initial check
            degs = adj.sum(axis=1)
            if np.min(degs) < target_deg:
                continue
            
            # Greedy removal
            # To speed up, we can break early if time is running out
            for step in range(3000):
                if time.time() > deadline:
                    return None
                
                # Compute common neighbors
                common = adj @ adj
                
                # Find violations: pairs (u, v) u!=v with common[u,v] >= 2
                # Mask diagonal
                # We can use np.triu_indices to get u < v
                # Or just argwhere and filter
                
                # Efficient check:
                # Create a copy of common, set diagonal to 0
                # Then check max > 1
                
                # We need to find ONE violation to fix.
                # We can just check if any exist.
                # To avoid full argwhere if possible?
                # But n is small.
                
                # Let's find all violations
                # mask = np.ones((n, n), dtype=bool)
                # np.fill_diagonal(mask, False)
                # violations = np.argwhere((common >= 2) & mask)
                
                # Optimization: 
                # If we just want to know if valid, we can check max.
                # But we need to fix.
                
                # Let's use a random sampling approach to find violations faster?
                # No, argwhere is fine for 46x46.
                
                # However, we can optimize by only checking upper triangle?
                # common is symmetric.
                
                # Let's just do:
                # tmp = common.copy()
                # np.fill_diagonal(tmp, 0)
                # if tmp.max() < 2: break
                
                # If max >= 2, find one.
                # We can pick a random element from the set of violations.
                # To get the set, we use argwhere.
                
                # Alternative: 
                # Pick a random pair (u, v) and check if it's a violation?
                # If not, try another?
                # This might be slower if few violations.
                # But if many violations, it's fast.
                
                # Let's stick to argwhere for reliability.
                
                tmp = common.copy()
                np.fill_diagonal(tmp, 0)
                if tmp.max() < 2:
                    # Valid C4-free
                    if np.min(adj.sum(axis=1)) >= target_deg:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        break # Degree too low
                
                # Find violations
                # We can restrict to upper triangle to avoid duplicates
                # u, v indices
                # Let's get all (u, v) with u < v
                # indices = np.triu_indices(n, k=1)
                # vals = tmp[indices]
                # viol_mask = vals >= 2
                # if not np.any(viol_mask): break
                # viol_indices = np.where(viol_mask)[0]
                # pick one
                # pos = random.choice(viol_indices)
                # u, v = indices[0][pos], indices[1][pos]
                
                # This is efficient.
                indices = np.triu_indices(n, k=1)
                vals = tmp[indices]
                viol_mask = vals >= 2
                if not np.any(viol_mask):
                    break
                viol_positions = np.where(viol_mask)[0]
                pos = random.choice(viol_positions)
                u, v = indices[0][pos], indices[1][pos]
                
                # Find common neighbors
                # w such that adj[u,w]==1 and adj[v,w]==1
                # We can use bitwise AND
                commons = np.where((adj[u] & adj[v]) == 1)[0]
                
                if len(commons) == 0:
                    # Should not happen
                    continue
                
                w = random.choice(commons.tolist())
                
                deg_u = adj[u].sum()
                deg_v = adj[v].sum()
                
                if deg_u > deg_v:
                    x = u
                elif deg_v > deg_u:
                    x = v
                else:
                    x = u if random.random() < 0.5 else v
                
                # Remove (x, w)
                adj[x][w] = 0
                adj[w][x] = 0
                
                # Check min degree
                if np.min(adj.sum(axis=1)) < target_deg:
                    break
        
        return None
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        for seed in range(300):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_result = res
                found = True
                if n == 47:
                    return best_result
                break
        if found:
            continue
        else:
            pass
    
    return best_result
