import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0
    
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    C = a @ b.T
    a_diag = np.diag(a).copy()
    
    def get_deltas_for_i(i, p):
        u = p[i]
        Ci_p = C[i, p]
        Ci_u = C[i, u]
        Cj_pj = C[np.arange(n), p]
        Cj_u = C[:, u]
        
        S = Ci_p - Ci_u - Cj_pj + Cj_u
        
        term_i = (a[i, i] - a[:, i]) * (b[p, i] - b[u, i])
        
        b_p_diag = b[p, np.arange(n)]
        term_j = (a[i, :] - a_diag) * (b_p_diag - b[u, :])
        
        deltas = S - term_i - term_j
        deltas[i] = np.inf
        return deltas

    # Initial
    row_ind, col_ind = linear_sum_assignment(C)
    p = col_ind.copy()
    current_cost = calculate_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    def tabu_search(p_init, time_budget):
        p = p_init.copy()
        current_cost = calculate_cost(p)
        best_p_local = p.copy()
        best_cost_local = current_cost
        
        tabu_tenure = 50
        max_no_improve = 50
        no_improve = 0
        iteration = 0
        tabu_list = {}
        
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() > end_time:
                    break
                deltas = get_deltas_for_i(i, p)
                min_idx = np.argmin(deltas)
                min_val = deltas[min_idx]
                
                if min_val < best_delta:
                    j = min_idx
                    key1 = (i, j)
                    key2 = (j, i)
                    is_tabu = (key1 in tabu_list and tabu_list[key1] > iteration) or \
                              (key2 in tabu_list and tabu_list[key2] > iteration)
                    
                    if not is_tabu or (current_cost + min_val < best_cost_local):
                        best_delta = min_val
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                tabu_list[(best_i, best_j)] = iteration + tabu_tenure
                tabu_list[(best_j, best_i)] = iteration + tabu_tenure
                
                if current_cost < best_cost_local:
                    best_cost_local = current_cost
                    best_p_local = p.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            iteration += 1
            if iteration % 100 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
                
            if no_improve > max_no_improve:
                break
                
        return best_p_local, best_cost_local

    # Main Loop
    p_sol, cost_sol = tabu_search(p, time_limit - 10.0)
    if cost_sol < best_cost:
        best_cost = cost_sol
        best_p = p_sol
        
    # Restart if time
    while time.time() - start_time < time_limit - 5.0:
        # Perturb
        p_new = best_p.copy()
        for _ in range(10):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                p_new[i], p_new[j] = p_new[j], p_new[i]
                
        remaining = time_limit - 5.0 - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        p_sol, cost_sol = tabu_search(p_new, remaining)
        if cost_sol < best_cost:
            best_cost = cost_sol
            best_p = p_sol
            
    return best_p.tolist()
