"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Hungarian-based rank structure
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    proxy_cost = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        old_row_i = np.dot(a[i, :], b[pi, p])
        old_col_i = np.dot(a[:, i], b[p, pi])
        old_row_j = np.dot(a[j, :], b[pj, p])
        old_col_j = np.dot(a[:, j], b[p, pj])
        
        old_ii = a[i, i] * b[pi, pi]
        old_ij = a[i, j] * b[pi, pj]
        old_ji = a[j, i] * b[pj, pi]
        old_jj = a[j, j] * b[pj, pj]
        
        old_cost_s = old_row_i + old_col_i + old_row_j + old_col_j - old_ii - old_ij - old_ji - old_jj
        
        v_nri = b[pj, p].copy()
        v_nri[i] = b[pj, pj]
        v_nri[j] = b[pj, pi]
        
        v_nci = b[p, pj].copy()
        v_nci[i] = b[pj, pj]
        v_nci[j] = b[pi, pj]
        
        v_nrj = b[pi, p].copy()
        v_nrj[i] = b[pi, pj]
        v_nrj[j] = b[pi, pi]
        
        v_ncj = b[p, pi].copy()
        v_ncj[i] = b[pj, pi]
        v_ncj[j] = b[pi, pi]
        
        new_row_i = np.dot(a[i, :], v_nri)
        new_col_i = np.dot(a[:, i], v_nci)
        new_row_j = np.dot(a[j, :], v_nrj)
        new_col_j = np.dot(a[:, j], v_ncj)
        
        new_ii = a[i, i] * b[pj, pj]
        new_ij = a[i, j] * b[pj, pi]
        new_ji = a[j, i] * b[pi, pj]
        new_jj = a[j, j] * b[pi, pi]
        
        new_cost_s = new_row_i + new_col_i + new_row_j + new_col_j - new_ii - new_ij - new_ji - new_jj
        
        return new_cost_s - old_cost_s

    def local_search(p, current_cost, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() >= time_limit:
                        break
                    delta = delta_swap(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
            if time.time() >= time_limit:
                break
        return p, current_cost

    start_time = time.time()
    time_limit = 118.0
    
    # Initial Local Search
    best_p, best_cost = local_search(p.copy(), calc_cost(p), time_limit)
    current_p = best_p.copy()
    current_cost = best_cost
    
    # ILS Loop
    strength = 5
    
    while time.time() - start_time < time_limit:
        # Perturb
        p_pert = current_p.copy()
        for _ in range(strength):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Local Search on Perturbed
        p_new, cost_new = local_search(p_pert, calc_cost(p_pert), time_limit)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new
            current_p = p_new
            current_cost = cost_new
            strength = 5 # Reset strength on improvement
        else:
            strength = min(strength + 1, 20)
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
