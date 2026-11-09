import numpy as np
import random
import time

def construct():
    target_n = 42
    random.seed(42)
    
    # Initialize random graph
    adj = np.zeros((target_n, target_n), dtype=np.int8)
    for i in range(target_n):
        for j in range(i+1, target_n):
            if random.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1

    def get_neighbors(v):
        return np.where(adj[v] == 1)[0]
        
    def get_non_neighbors(v):
        non_neigh = np.where(adj[v] == 0)[0]
        return non_neigh[non_neigh != v]

    def count_triangles_bitmask(sub_indices, adj_matrix):
        if len(sub_indices) < 3:
            return 0
        k = len(sub_indices)
        local_adj = [0] * k
        for i in range(k):
            u = sub_indices[i]
            for j in range(i+1, k):
                v = sub_indices[j]
                if adj_matrix[u, v]:
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
                    
        count = 0
        for i in range(k):
            for j in range(i+1, k):
                if local_adj[i] & (1 << j):
                    common = local_adj[i] & local_adj[j]
                    count += bin(common).count('1')
        return count // 3

    def count_independent_triangles_bitmask(sub_indices, adj_matrix):
        if len(sub_indices) < 3:
            return 0
        k = len(sub_indices)
        local_adj = [0] * k
        for i in range(k):
            u = sub_indices[i]
            for j in range(i+1, k):
                v = sub_indices[j]
                if not adj_matrix[u, v]: # Complement
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
                    
        count = 0
        for i in range(k):
            for j in range(i+1, k):
                if local_adj[i] & (1 << j):
                    common = local_adj[i] & local_adj[j]
                    count += bin(common).count('1')
        return count // 3

    from itertools import combinations
    def full_cost_check(adj, n):
        k5 = 0
        ind5 = 0
        for combo in combinations(range(n), 5):
            is_c = True
            is_i = True
            for idx in range(5):
                for jdx in range(idx+1, 5):
                    u, v = combo[idx], combo[jdx]
                    if adj[u, v]:
                        is_i = False
                    else:
                        is_c = False
                    if not is_c and not is_i:
                        break
                if not is_c and not is_i:
                    break
            if is_c: k5 += 1
            if is_i: ind5 += 1
        return k5 + ind5

    # Initial cost
    current_cost = full_cost_check(adj, target_n)
    best_cost = current_cost
    best_adj = adj.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_steps = 50000
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > 250:
            break
            
        u = random.randint(0, target_n - 1)
        v = random.randint(0, target_n - 1)
        if u == v:
            continue
            
        common_neigh = np.intersect1d(get_neighbors(u), get_neighbors(v))
        common_neigh = common_neigh[(common_neigh != u) & (common_neigh != v)]
        
        common_non_neigh = np.intersect1d(get_non_neighbors(u), get_non_neighbors(v))
        common_non_neigh = common_non_neigh[(common_non_neigh != u) & (common_non_neigh != v)]
        
        num_k5_potential = count_triangles_bitmask(common_neigh, adj)
        num_ind5_potential = count_independent_triangles_bitmask(common_non_neigh, adj)
        
        if adj[u, v] == 0:
            delta = num_k5_potential - num_ind5_potential
        else:
            delta = -num_k5_potential + num_ind5_potential
            
        if delta < 0:
            accept = True
        else:
            r = random.random()
            if r < np.exp(-delta / T):
                accept = True
            else:
                accept = False
                
        if accept:
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = adj.copy()
                
            if current_cost == 0:
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if best_adj[i, j]:
                            edges.append((i, j))
                return target_n, edges
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_cost == 0:
        edges = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                if best_adj[i, j]:
                    edges.append((i, j))
        return target_n, edges
        
    # Fallback to Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj_paley = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj_paley[i, j] = 1
                adj_paley[j, i] = 1
                
    edges = []
    for i in range(q):
        for j in range(i+1, q):
            if adj_paley[i, j]:
                edges.append((i, j))
                
    return q, edges
