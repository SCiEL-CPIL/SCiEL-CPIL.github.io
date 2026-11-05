import numpy as np
import time

def construct():
    # Fallback seed
    seed_n = 11
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    
    # Time allocation
    # We have 120s limit. Use 110s.
    total_time = 110.0
    start_total = time.time()
    
    # Try n=26
    t_start = time.time()
    res = solve_n(26, time_limit=80.0)
    if res is not None:
        return res
        
    # Try n=25
    elapsed = time.time() - start_total
    remaining = total_time - elapsed - 5.0 # 5s margin
    if remaining > 10:
        res = solve_n(25, time_limit=remaining)
        if res is not None:
            return res
            
    # Try n=24
    elapsed = time.time() - start_total
    remaining = total_time - elapsed - 5.0
    if remaining > 5:
        res = solve_n(24, time_limit=remaining)
        if res is not None:
            return res
            
    return {'n': seed_n, 'colors': seed_colors}

def solve_n(n, time_limit):
    start_time = time.time()
    
    edge_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_indices)
    
    colors = np.zeros((n, n), dtype=np.int8)
    Ms = [np.zeros((n, n), dtype=np.float32) for _ in range(5)]
    
    def count_c4s(M):
        M2 = M @ M
        M4 = M2 @ M2
        return np.trace(M4) / 8.0
        
    # Greedy Init
    np.random.seed(42)
    edge_order = np.random.permutation(num_edges)
    
    for idx in edge_order:
        i, j = edge_indices[idx]
        
        current_costs = [count_c4s(M) for M in Ms]
        
        best_c = 0
        min_increase = np.inf
        
        for c in range(5):
            Ms[c][i, j] = 1
            Ms[c][j, i] = 1
            
            new_cost = count_c4s(Ms[c])
            increase = new_cost - current_costs[c]
            
            Ms[c][i, j] = 0
            Ms[c][j, i] = 0
            
            if increase < min_increase:
                min_increase = increase
                best_c = c
                
        colors[i, j] = best_c
        colors[j, i] = best_c
        Ms[best_c][i, j] = 1
        Ms[best_c][j, i] = 1
        
    current_cost = sum(count_c4s(M) for M in Ms)
    best_cost = current_cost
    best_colors = colors.copy()
    
    if current_cost == 0:
        return {'n': n, 'colors': [int(best_colors[i, j]) for i in range(n) for j in range(i + 1, n)]}
        
    # SA
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() - start_time < time_limit:
        idx = np.random.randint(num_edges)
        i, j = edge_indices[idx]
        
        old_c = colors[i, j]
        new_c = np.random.randint(5)
        if new_c == old_c:
            continue
            
        Ms[old_c][i, j] = 0
        Ms[old_c][j, i] = 0
        Ms[new_c][i, j] = 1
        Ms[new_c][j, i] = 1
        
        new_cost = sum(count_c4s(M) for M in Ms)
        
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_cost = new_cost
            colors[i, j] = new_c
            colors[j, i] = new_c
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return {'n': n, 'colors': [int(best_colors[i, j]) for i in range(n) for j in range(i + 1, n)]}
        else:
            Ms[new_c][i, j] = 0
            Ms[new_c][j, i] = 0
            Ms[old_c][i, j] = 1
            Ms[old_c][j, i] = 1
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    # If we didn't find 0, return None (invalid)
    # We must return a valid coloring.
    # If best_cost > 0, the coloring is invalid.
    # So we return None to signal failure.
    return None
