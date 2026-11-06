import time
import random
import itertools

def construct():
    deadline = time.time() + 100.0
    best_result = None
    
    # Try to find a graph on n=35, then 34, etc.
    # We know n=35 is the max possible.
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
        # Try to find a valid graph on n vertices
        result = _search_for_graph(n, deadline)
        if result is not None:
            return result
            
    # Fallback to seed if all else fails (should not happen if seed is valid)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _search_for_graph(n, deadline):
    # Try multiple random restarts
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        # Initialize random graph
        # Use a density that might be close to valid graphs.
        # For R(4,6), density is roughly 0.5?
        adj = [[False]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append((i, j))
        
        # Local search
        for step in range(1000):
            if time.time() > deadline:
                return None
            
            # Check for K4
            k4 = _find_k4(adj, n)
            if k4 is not None:
                # Remove a random edge from the K4
                u, v, w, x = k4
                # Pick a random pair from the 6 edges
                pair_idx = random.randint(0, 5)
                pairs = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                a, b = pairs[pair_idx]
                adj[a][b] = False
                adj[b][a] = False
                edges.remove((a, b))
                continue
            
            # Check for Independent Set of size 6
            ind6 = _find_ind_set(adj, n, 6)
            if ind6 is not None:
                # Add a random edge within the independent set
                # Pick two random vertices from the set
                v1, v2 = random.sample(ind6, 2)
                if not adj[v1][v2]:
                    adj[v1][v2] = True
                    adj[v2][v1] = True
                    edges.append((v1, v2))
                continue
            
            # If no K4 and no Ind-6, it's valid
            return {'n': n, 'edges': [list(e) for e in edges]}
            
        # If loop finishes without finding valid, restart
    return None

def _find_k4(adj, n):
    # Find any K4
    # Iterate over all quadruples
    for i in range(n-3):
        for j in range(i+1, n-2):
            if not adj[i][j]: continue
            for k in range(j+1, n-1):
                if not adj[i][k] or not adj[j][k]: continue
                for l in range(k+1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return (i, j, k, l)
    return None

def _find_ind_set(adj, n, size):
    # Find any independent set of given size
    # Backtracking search
    # To make it faster, order vertices by degree?
    # For simplicity, just DFS
    
    def dfs(current_set, start_idx):
        if len(current_set) == size:
            return current_set
        # Pruning: if not enough vertices left
        if start_idx + (size - len(current_set)) > n:
            return None
        
        for v in range(start_idx, n):
            # Check if v is independent from all in current_set
            is_ind = True
            for u in current_set:
                if adj[u][v]:
                    is_ind = False
                    break
            if is_ind:
                res = dfs(current_set + [v], v + 1)
                if res is not None:
                    return res
        return None
        
    return dfs([], 0)
