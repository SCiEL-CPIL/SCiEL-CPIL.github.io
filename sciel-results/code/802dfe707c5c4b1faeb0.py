import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Seed solution
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': colors_seed}
    
    # Try n from 18 down to 10
    # 18 is the target. 17 is fallback.
    
    for n in range(18, 9, -1):
        if time.time() > deadline - 10:
            break
            
        found = solve_n(n, deadline)
        if found:
            best_result = found
            if n >= 17:
                # If we found 17 or 18, it's a good result.
                # We can break if we found 18.
                # If we found 17, we might want to try 18 again? 
                # The loop is descending, so 18 was tried first.
                # If 18 failed, we tried 17.
                # If 17 succeeded, we return it.
                break
            # If we found 16, we continue to try 17? 
            # No, loop is 18, 17, 16...
            # So if 18 fails, 17 is tried. If 17 fails, 16 is tried.
            # We return the first success.
            
    return best_result

def solve_n(n, deadline):
    m = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Random init
    colors = np.random.randint(0, 4, size=m)
    
    # Precompute edge indices for faster access?
    # colors is a flat array.
    
    cost = calculate_cost(n, colors)
    
    temp = 5.0
    cooling = 0.995
    min_temp = 0.01
    
    # We run SA for a fixed number of iterations or until deadline
    iter_limit = 50000
    iters = 0
    
    while iters < iter_limit and time.time() < deadline - 5:
        iters += 1
        idx = random.randint(0, m - 1)
        old_c = colors[idx]
        new_c = (old_c + 1 + random.randint(0, 2)) % 4
        
        if new_c == old_c:
            continue
            
        colors[idx] = new_c
        new_cost = calculate_cost(n, colors)
        delta = new_cost - cost
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            cost = new_cost
            if cost < 100: # Threshold for exact check
                if not has_mono_c6(n, colors, edges):
                    return {'n': n, 'colors': colors.tolist()}
        else:
            colors[idx] = old_c
            
        temp = max(min_temp, temp * cooling)
        
    return None

def calculate_cost(n, colors):
    # Sum of trace(A_c^6)
    # This is a proxy for number of 6-walks
    total = 0
    for c in range(4):
        A = np.zeros((n, n), dtype=np.int64)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == c:
                    A[i, j] = 1
                    A[j, i] = 1
                idx += 1
        # A^6
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        total += np.trace(A6)
    return total

def has_mono_c6(n, colors, edges):
    # Build adj lists
    adj = [set() for _ in range(4)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj[c].add((i, j))
        adj[c].add((j, i))
        
    # Convert to list of neighbors for DFS
    adj_list = [[[] for _ in range(n)] for _ in range(4)]
    for c in range(4):
        for (u, v) in adj[c]:
            adj_list[c][u].append(v)
            
    for c in range(4):
        if has_c6(adj_list[c], n):
            return True
    return False

def has_c6(adj_list, n):
    # Check for C6
    for u in range(n):
        if dfs_c6(adj_list, u, u, 0, {u}):
            return True
    return False

def dfs_c6(adj_list, start, current, depth, visited):
    if depth == 6:
        return current == start
    if current not in adj_list or not adj_list[current]:
        return False
    for nb in adj_list[current]:
        if nb == start:
            if depth == 5:
                return True
            continue
        if nb not in visited:
            visited.add(nb)
            if dfs_c6(adj_list, start, nb, depth + 1, visited):
                return True
            visited.remove(nb)
    return False
