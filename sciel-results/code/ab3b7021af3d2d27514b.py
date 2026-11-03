import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initial heuristic: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def get_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    timeout = 115.0 # Safety margin
    
    # Tabu parameters
    tabu_tenure = 25
    tabu_list = [] # List of (i, j)
    tabu_set = set()
    
    # Precompute nothing, compute on fly
    
    while time.time() - start_time < timeout:
        best_delta = 2**63 # Large int
        best_i = -1
        best_j = -1
        best_is_tabu = False
        
        # Iterate over all pairs i < j
        for i in range(n):
            u = p[i]
            # Precompute rows/cols for i to reuse?
            # a_i = a[i, :]
            # a_col_i = a[:, i]
            # b_u = b[u, :]
            # b_col_u = b[:, u]
            # But these depend on u, which is fixed for i in this inner loop.
            a_i = a[i, :]
            a_col_i = a[:, i]
            b_u = b[u, :]
            b_col_u = b[:, u]
            
            for j in range(i + 1, n):
                v = p[j]
                
                # Check tabu
                is_tabu = (i, j) in tabu_set
                
                # Compute delta
                a_j = a[j, :]
                a_col_j = a[:, j]
                b_v = b[v, :]
                b_col_v = b[:, v]
                
                # da = a_i - a_j
                # dat = a_col_i - a_col_j
                # bv_minus_bu = b_v - b_u
                # bcv_minus_bcu = b_col_v - b_col_u
                
                total_sum = np.dot(a_i - a_j, b_v - b_u) + np.dot(a_col_i - a_col_j, b_col_v - b_col_u)
                
                ti = (a[i, i] - a[j, i]) * (b[v, i] - b[u, i]) + (a[i, i] - a[j, i]) * (b[i, v] - b[i, u])
                tj = (a[i, j] - a[j, j]) * (b[v, j] - b[u, j]) + (a[i, j] - a[j, j]) * (b[j, v] - b[j, u])
                
                sum_part = total_sum - ti - tj
                const_part = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v]) + (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
                
                delta = sum_part + const_part
                
                # Aspiration: accept if not tabu, or if it improves best_cost
                if not is_tabu:
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        best_is_tabu = False
                else:
                    new_cost = current_cost + delta
                    if new_cost < best_cost:
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                            best_is_tabu = True
        
        # If no move found, break
        if best_i == -1:
            break
            
        # Perform the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu List
        tabu_list.append((best_i, best_j))
        tabu_set.add((best_i, best_j))
        
        if len(tabu_list) > tabu_tenure:
            old_move = tabu_list.pop(0)
            tabu_set.remove(old_move)
            
    return best_p.tolist()
