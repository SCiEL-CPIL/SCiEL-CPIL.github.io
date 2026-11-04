"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        # Optimized delta calculation avoiding array copies
        # Old costs
        old_row_i = np.dot(a[i, :], b[pi, p])
        old_col_i = np.dot(a[:, i], b[p, pi])
        old_row_j = np.dot(a[j, :], b[pj, p])
        old_col_j = np.dot(a[:, j], b[p, pj])
        
        old_diag_corr = a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
        old_affected = old_row_i + old_col_i + old_row_j + old_col_j - old_diag_corr
        
        # New costs
        # new_row_i = sum_v a[i,v] b[pj, p_new[v]]
        # p_new[i]=pj, p_new[j]=pi, p_new[k]=p[k]
        # b[pj, p] has b[pj, p[i]]=b[pj, pi] at idx i, b[pj, p[j]]=b[pj, pj] at idx j
        # We need b[pj, pj] at idx i, b[pj, pi] at idx j
        # So new_row_i = dot(a[i,:], b[pj,p]) - a[i,i]*b[pj,pi] - a[i,j]*b[pj,pj] + a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi]
        base_row_i = np.dot(a[i, :], b[pj, p])
        new_row_i = base_row_i - a[i,i]*b[pj,pi] - a[i,j]*b[pj,pj] + a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi]
        
        # new_col_i = sum_u a[u,i] b[p_new[u], pj]
        # b[p, pj] has b[p[i], pj]=b[pi, pj] at idx i, b[p[j], pj]=b[pj, pj] at idx j
        # We need b[pj, pj] at idx i, b[pi, pj] at idx j
        base_col_i = np.dot(a[:, i], b[p, pj])
        new_col_i = base_col_i - a[i,i]*b[pi,pj] - a[j,i]*b[pj,pj] + a[i,i]*b[pj,pj] + a[j,i]*b[pi,pj]
        
        # new_row_j = sum_v a[j,v] b[pi, p_new[v]]
        # b[pi, p] has b[pi, p[i]]=b[pi, pi] at idx i, b[pi, p[j]]=b[pi, pj] at idx j
        # We need b[pi, pj] at idx i, b[pi, pi] at idx j
        base_row_j = np.dot(a[j, :], b[pi, p])
        new_row_j = base_row_j - a[j,i]*b[pi,pi] - a[j,j]*b[pi,pj] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
        
        # new_col_j = sum_u a[u,j] b[p_new[u], pi]
        # b[p, pi] has b[p[i], pi]=b[pi, pi] at idx i, b[p[j], pi]=b[pj, pi] at idx j
        # We need b[pj, pi] at idx i, b[pi, pi] at idx j
        base_col_j = np.dot(a[:, j], b[p, pi])
        new_col_j = base_col_j - a[i,j]*b[pi,pi] - a[j,j]*b[pj,pi] + a[i,j]*b[pj,pi] + a[j,j]*b[pi,pi]
        
        new_diag_corr = a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
        
        new_affected = new_row_i + new_col_i + new_row_j + new_col_j - new_diag_corr
        
        return new_affected - old_affected
    
    # 1. Initialization: Enhanced Rank-Based Proxy
    # Use combined row and column sums to capture total interaction strength.
    # This better reflects the "global rank structure" of the matrices.
    a_strength = a.sum(axis=1) + a.sum(axis=0)
    b_strength = b.sum(axis=1) + b.sum(axis=0)
    
    # We want high a_strength to match low b_strength to minimize cost.
    # Proxy cost = a_strength[i] * b_strength[j]
    proxy_cost = np.outer(a_strength, b_strength)
    
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Quick local search polish on the initial solution to fix obvious bad swaps
    # This ensures we start SA from a much better local minimum.
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Randomize order to avoid bias, or just standard sweep
        for i in range(n):
            for j in range(i + 1, n):
                delta = delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    # 2. Simulated Annealing with Restart
    # We will run multiple SA phases, restarting from best solution if stuck.
    # Adjusted parameters for better balance between exploration and exploitation.
    
    T_start = 500.0
    T_end = 0.5
    cooling_rate = 0.9999
    max_no_improve = 3000
    
    while time.time() - start_time < time_limit:
        T = T_start
        no_improve_counter = 0
        
        while T > T_end and time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            delta = delta_swap(p, i, j)
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            
            T *= cooling_rate
            
            if no_improve_counter > max_no_improve:
                break
        
        # If we broke out due to no improvement, restart from best
        if no_improve_counter > max_no_improve and time.time() - start_time < time_limit:
            p = best_p.copy()
            current_cost = best_cost
    
    # 3. Final Local Search Polish
    # Try all pairs for improvement.
    # Use a random order in the inner loop to escape local minima faster?
    # No, standard best-improvement is safer for final polish.
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
