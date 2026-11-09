import time
import random
import numpy as np
from collections import defaultdict

N = 64

def _sq_dist(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def _is_valid_with_set(p, S):
    """Check if point p can be added to set S without creating an isosceles triangle."""
    if not S:
        return True
    
    # Distances from p to each point in S
    d_to_S = [_sq_dist(p, s) for s in S]
    
    # Check 1: all distances from p to S must be distinct
    if len(set(d_to_S)) != len(d_to_S):
        return False
    
    # Check 2: for each pair (a,b) in S, the distance d(a,b) must not equal d(p,a) or d(p,b)
    # Equivalently: no d(p,a) should equal d(a,b) for any b in S, b != a
    # And no d(p,b) should equal d(a,b) for any a in S, a != b
    # This means: for each a in S, d(p,a) must not be in {d(a,b) : b in S, b != a}
    
    n = len(S)
    for i in range(n):
        dpi = d_to_S[i]
        # Check if dpi equals distance from S[i] to any other point in S
        for j in range(n):
            if i == j:
                continue
            if _sq_dist(S[i], S[j]) == dpi:
                return False
    
    return True

def _greedy_construct(dist_mat, order, n_points, deadline):
    """Fast greedy construction using precomputed distance matrix."""
    S = []
    S_set = set()
    
    for p in order:
        if time.time() > deadline:
            break
        
        if p in S_set:
            continue
        
        if not S:
            S.append(p)
            S_set.add(p)
            continue
        
        # Check validity using distance matrix
        S_arr = np.array(S, dtype=int)
        d_to_S = dist_mat[p, S_arr]
        
        # Check 1: unique distances from p to S
        d_sorted = np.sort(d_to_S)
        if np.any(d_sorted[1:] == d_sorted[:-1]):
            continue
        
        # Check 2: no conflict with existing distances
        D_S = dist_mat[np.ix_(S_arr, S_arr)]
        valid = True
        for j in range(len(S)):
            if d_to_S[j] in D_S[j, :]:
                valid = False
                break
            if d_to_S[j] in D_S[:, j]:
                valid = False
                break
        
        if valid:
            S.append(p)
            S_set.add(p)
    
    return S

def _local_search(S, dist_mat, n_points, deadline, max_iter=5000):
    """Local search: try to add points, and if stuck, try swaps."""
    S = S.copy()
    S_set = set(S)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to add any valid point
        added = False
        for p in range(n_points):
            if p in S_set:
                continue
            
            S_arr = np.array(S, dtype=int)
            d_to_S = dist_mat[p, S_arr]
            
            d_sorted = np.sort(d_to_S)
            if np.any(d_sorted[1:] == d_sorted[:-1]):
                continue
            
            D_S = dist_mat[np.ix_(S_arr, S_arr)]
            valid = True
            for j in range(len(S)):
                if d_to_S[j] in D_S[j, :]:
                    valid = False
                    break
                if d_to_S[j] in D_S[:, j]:
                    valid = False
                    break
            
            if valid:
                S.append(p)
                S_set.add(p)
                added = True
                break
        
        if added:
            continue
        
        # Try swap: remove one point, try to add one or two others
        swap_done = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            
            S_without_i = [S[j] for j in range(len(S)) if j != i]
            S_without_i_set = set(S_without_i)
            
            # Try adding each point not in S_without_i
            for p in range(n_points):
                if p in S_without_i_set:
                    continue
                
                S_arr = np.array(S_without_i, dtype=int)
                if len(S_arr) == 0:
                    S = [p]
                    S_set = {p}
                    swap_done = True
                    break
                
                d_to_S = dist_mat[p, S_arr]
                d_sorted = np.sort(d_to_S)
                if np.any(d_sorted[1:] == d_sorted[:-1]):
                    continue
                
                D_S = dist_mat[np.ix_(S_arr, S_arr)]
                valid = True
                for j in range(len(S_without_i)):
                    if d_to_S[j] in D_S[j, :]:
                        valid = False
                        break
                    if d_to_S[j] in D_S[:, j]:
                        valid = False
                        break
                
                if valid:
                    S = S_without_i + [p]
                    S_set = set(S)
                    swap_done = True
                    break
            
            if swap_done:
                break
        
        if not swap_done:
            break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_points = len(all_points)
    
    # Precompute squared distance matrix
    coords = np.array(all_points, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sum(diff * diff, axis=2).astype(np.int64)
    
    best_set = []
    
    # Phase 1: Multiple greedy restarts to get good initial solutions
    n_greedy = 200
    greedy_solutions = []
    
    for restart in range(n_greedy):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        order = list(range(n_points))
        random.shuffle(order)
        
        S = _greedy_construct(dist_mat, order, n_points, deadline)
        
        if len(S) > len(best_set):
            best_set = S.copy()
        
        greedy_solutions.append(S)
    
    # Phase 2: Local search on top of best greedy solutions
    # Sort greedy solutions by size, take top ones for local search
    greedy_solutions.sort(key=len, reverse=True)
    top_k = min(20, len(greedy_solutions))
    
    for i in range(top_k):
        if time.time() > deadline:
            break
        
        S = greedy_solutions[i].copy()
        
        # Run local search with different random seeds for diversity
        for ls_seed in range(5):
            if time.time() > deadline:
                break
            
            random.seed(1000 + i * 10 + ls_seed)
            S_ls = _local_search(S, dist_mat, n_points, deadline, max_iter=3000)
            
            if len(S_ls) > len(best_set):
                best_set = S_ls.copy()
        
        # If we already have a very good solution, we can stop early
        if len(best_set) >= 60:
            break
    
    # Phase 3: Final greedy passes with different orderings if time remains
    if time.time() < deadline - 30:
        for restart in range(n_greedy, 500):
            if time.time() > deadline:
                break
            
            random.seed(restart)
            order = list(range(n_points))
            random.shuffle(order)
            
            S = _greedy_construct(dist_mat, order, n_points, deadline)
            
            if len(S) > len(best_set):
                best_set = S.copy()
    
    return [all_points[i] for i in best_set]
