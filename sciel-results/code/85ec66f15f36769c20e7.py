import time
import random
import numpy as np

def _get_idx(i, j, n):
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _validate_coloring(n, colors):
    for c in range(5):
        A = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == c:
                    A[i, j] = 1
                    A[j, i] = 1
                idx += 1
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
    return True

def _backtracking_search(n, deadline, seed=42):
    rng = random.Random(seed)
    n_edges = n * (n - 1) // 2
    colors = np.full(n_edges, -1, dtype=np.int8)
    common = np.zeros((5, n, n), dtype=np.int8)
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute neighbors for each edge?
    # For edge (i,j), the affected pairs are (i,k) and (j,k).
    # We need idx(i,k) and idx(j,k).
    
    def dfs(edge_idx, colors, common):
        if time.time() > deadline:
            return False
        
        if edge_idx == n_edges:
            return True
        
        i, j = edges[edge_idx]
        
        valid_colors = []
        for c in range(5):
            if common[c][i][j] >= 2:
                continue
                
            is_valid = True
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_jk = _get_idx(j, k, n)
                idx_ik = _get_idx(i, k, n)
                
                if colors[idx_jk] == c:
                    if common[c][i][k] >= 1:
                        is_valid = False
                        break
                if colors[idx_ik] == c:
                    if common[c][j][k] >= 1:
                        is_valid = False
                        break
            
            if is_valid:
                valid_colors.append(c)
                
        if not valid_colors:
            return False
            
        rng.shuffle(valid_colors)
        
        for c in valid_colors:
            colors[edge_idx] = c
            new_common = common.copy()
            
            for k in range(n):
                if k == i or k == j:
                    continue
                idx_jk = _get_idx(j, k, n)
                idx_ik = _get_idx(i, k, n)
                
                if colors[idx_jk] == c:
                    new_common[c][i][k] += 1
                    new_common[c][k][i] += 1
                if colors[idx_ik] == c:
                    new_common[c][j][k] += 1
                    new_common[c][k][j] += 1
            
            if dfs(edge_idx + 1, colors, new_common):
                return True
            
            colors[edge_idx] = -1
            
        return False

    if dfs(0, colors, common):
        return colors.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 1
    best_colors = [0]
    
    # Try to find a large n first
    # If we fail, we fall back to smaller n.
    # We try n=26 down to 15.
    # This is a "maximize n" strategy, so we start high.
    
    for n in range(26, 14, -1):
        if time.time() > deadline - 15:
            break
            
        found = None
        # Try a few seeds
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline - 5:
                break
            result = _backtracking_search(n, deadline - 2, seed)
            if result is not None:
                if _validate_coloring(n, result):
                    found = result
                    break
        
        if found is not None:
            best_n = n
            best_colors = found
            break # Found the max n, stop
        
        # If we fail for n, we continue to n-1.
        # But if we fail for 26, 25 might work?
        # Usually if 26 is hard, 25 is easier.
        # So we keep looping downwards.
        
    # If we didn't find anything >= 15, try smaller
    if best_n == 1:
        for n in range(14, 0, -1):
            if time.time() > deadline - 5:
                break
            if n <= 3:
                best_n = n
                best_colors = [0] * (n * (n - 1) // 2)
                break
            result = _backtracking_search(n, deadline - 2, 42)
            if result is not None and _validate_coloring(n, result):
                best_n = n
                best_colors = result
                break

    return {'n': best_n, 'colors': best_colors}
