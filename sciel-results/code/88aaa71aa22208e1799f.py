import random
import time
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 140.0  # safety margin before 150s kill

def _score(adj, deg):
    """Compute max degree sum over all triangles using vectorized numpy ops where possible."""
    # adj is a numpy boolean matrix NxN
    # deg is a numpy int array of length N
    # Triangles: for each edge (i,j), common neighbors are where adj[i] & adj[j] is True
    # But we need i<j<k to avoid double counting, or just compute max over all triangles
    # For efficiency, we can iterate over edges and check common neighbors
    
    best = 0
    # Get all edges
    i_idx, j_idx = np.where(adj)
    # Only consider i < j to avoid duplicates
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    for i, j in zip(i_idx, j_idx):
        # Common neighbors of i and j
        common = np.where(adj[i] & adj[j])[0]
        for k in common:
            if k > j:  # ensure i < j < k
                s = deg[i] + deg[j] + deg[k]
                if s > best:
                    best = s
    return best

def construct():
    t0 = time.time()
    global_best_score = float('inf')
    global_best_adj = None
    
    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Run multiple restarts with different seeds
    for seed in range(20):
        if time.time() - t0 > DEADLINE:
            break
            
        rng = random.Random(seed * 7919 + 13)
        
        # Initialize: K_{8,8} + one internal edge (as in inspirations)
        adj = np.zeros((N, N), dtype=bool)
        for i in range(8):
            for j in range(8, 16):
                adj[i, j] = adj[j, i] = True
        adj[0, 1] = adj[1, 0] = True  # internal edge in part A
        
        deg = np.array([adj[i].sum() for i in range(N)])
        
        cur_score = _score(adj, deg)
        best_adj = adj.copy()
        best_deg = deg.copy()
        best_score = cur_score
        
        # Maintain edge and non-edge lists for efficient sampling
        edges = []
        nonedges = []
        for i, j in all_pairs:
            if adj[i, j]:
                edges.append((i, j))
            else:
                nonedges.append((i, j))
        
        # Simulated annealing parameters
        max_iters = 150000
        start_temp = 5.0
        end_temp = 0.1
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        
        temp = start_temp
        iters_done = 0
        
        while iters_done < max_iters:
            elapsed = time.time() - t0
            if elapsed > DEADLINE:
                break
            
            # Pick random edge to remove and non-edge to add
            e_remove = edges[rng.randrange(len(edges))]
            e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            # Perform swap
            adj[u, v] = adj[v, u] = False
            deg[u] -= 1
            deg[v] -= 1
            adj[x, y] = adj[y, x] = True
            deg[x] += 1
            deg[y] += 1
            
            new_score = _score(adj, deg)
            
            # Decide whether to accept
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.5
            else:
                delta = new_score - cur_score
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
                    best_deg = deg.copy()
                
                # Update edge/non-edge lists
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
            else:
                # Undo swap
                adj[x, y] = adj[y, x] = False
                deg[x] -= 1
                deg[y] -= 1
                adj[u, v] = adj[v, u] = True
                deg[u] += 1
                deg[v] += 1
            
            iters_done += 1
            temp *= cooling_rate
            
            # Check global best
            if best_score < global_best_score:
                global_best_score = best_score
                global_best_adj = best_adj.copy()
        
        # Early termination if we've found a very good solution
        if global_best_score <= 24:
            break
    
    # Final polish: do a local search from the best solution found
    if global_best_adj is not None and time.time() - t0 < DEADLINE - 10:
        adj = global_best_adj.copy()
        deg = np.array([adj[i].sum() for i in range(N)])
        cur_score = _score(adj, deg)
        
        # Local search: try all single swaps that improve the score
        improved = True
        while improved and time.time() - t0 < DEADLINE - 5:
            improved = False
            edges = [(i, j) for i, j in all_pairs if adj[i, j]]
            nonedges = [(i, j) for i, j in all_pairs if not adj[i, j]]
            
            for e_remove in edges:
                if time.time() - t0 > DEADLINE - 5:
                    break
                for e_add in nonedges:
                    u, v = e_remove
                    x, y = e_add
                    
                    # Perform swap
                    adj[u, v] = adj[v, u] = False
                    deg[u] -= 1
                    deg[v] -= 1
                    adj[x, y] = adj[y, x] = True
                    deg[x] += 1
                    deg[y] += 1
                    
                    new_score = _score(adj, deg)
                    
                    # Undo swap
                    adj[x, y] = adj[y, x] = False
                    deg[x] -= 1
                    deg[y] -= 1
                    adj[u, v] = adj[v, u] = True
                    deg[u] += 1
                    deg[v] += 1
                    
                    if new_score < cur_score:
                        # Accept the improvement
                        adj[u, v] = adj[v, u] = False
                        deg[u] -= 1
                        deg[v] -= 1
                        adj[x, y] = adj[y, x] = True
                        deg[x] += 1
                        deg[y] += 1
                        cur_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if cur_score < global_best_score:
            global_best_score = cur_score
            global_best_adj = adj.copy()
    
    # Return the best solution
    best_edges = [[i, j] for i, j in all_pairs if global_best_adj[i, j]]
    return {'n': N, 'edges': best_edges}
