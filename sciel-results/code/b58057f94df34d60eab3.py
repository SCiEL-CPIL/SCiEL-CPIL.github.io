import random
import time

def construct():
    n = 16
    # Try 13, 14, 15
    for K in [13, 14, 15]:
        result = _solve(K)
        if result is not None:
            return result
            
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _solve(K):
    n = 16
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    k4s.append((a, b, c, d))
                    
    num_k4s = len(k4s)
    
    edge_idx = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            edge_list.append((i, j))
            idx += 1
            
    k4_edges = []
    edge_to_k4s = [[] for _ in range(120)]
    
    for k_idx, (a, b, c, d) in enumerate(k4s):
        edges = [
            edge_idx[a][b],
            edge_idx[a][c],
            edge_idx[a][d],
            edge_idx[b][c],
            edge_idx[b][d],
            edge_idx[c][d]
        ]
        k4_edges.append(edges)
        for e in edges:
            edge_to_k4s[e].append(k_idx)
            
    colors = [random.randint(0, K-1) for _ in range(120)]
    
    def k4_cost(edge_indices, col_array):
        c0 = col_array[edge_indices[0]]
        c1 = col_array[edge_indices[1]]
        c2 = col_array[edge_indices[2]]
        c3 = col_array[edge_indices[3]]
        c4 = col_array[edge_indices[4]]
        c5 = col_array[edge_indices[5]]
        # Fast distinct count for 6 items
        # Using set is fine
        return max(0, 5 - len({c0, c1, c2, c3, c4, c5}))
        
    def total_cost(col_array):
        cost = 0
        for e_list in k4_edges:
            cost += k4_cost(e_list, col_array)
        return cost
        
    start_time = time.time()
    time_limit = 30.0 # Try 30s per K
    
    current_cost = total_cost(colors)
    best_colors = colors.copy()
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        e_idx = random.randint(0, 119)
        old_color = colors[e_idx]
        new_color = random.randint(0, K-1)
        
        if new_color == old_color:
            continue
            
        delta = 0
        for k_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k_idx]
            old_c = k4_cost(e_list, colors)
            
            colors[e_idx] = new_color
            new_c = k4_cost(e_list, colors)
            
            delta += new_c - old_c
            
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            colors[e_idx] = new_color
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors
        else:
            colors[e_idx] = old_color
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    if best_cost == 0:
        return best_colors
    return None
