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
        
        # Optimized delta calculation
        # Cost = sum_k sum_l a[k,l] b[p[k], p[l]]
        # Change p[i] <-> p[j]
        # Terms involving i or j change.
        
        # Let's use the efficient formula derived earlier but check for correctness.
        # The previous code's delta_swap was complex. Let's use a simpler but robust one.
        # Actually, for n=150, the previous delta_swap was probably fine.
        # Let's try to optimize it by precomputing some terms if possible?
        # No, p changes.
        
        # Alternative: Just calculate cost difference directly for the affected terms.
        # Affected terms: (i,i), (i,j), (j,i), (j,j) and all (k,i), (k,j), (i,k), (j,k) for k != i,j.
        
        # sum_k a[k,i] b[p[k], p[i]] -> p[i] becomes pj.
        # sum_k a[k,j] b[p[k], p[j]] -> p[j] becomes pi.
        # sum_k a[i,k] b[p[i], p[k]] -> p[i] becomes pj.
        # sum_k a[j,k] b[p[j], p[k]] -> p[j] becomes pi.
        
        # Let's compute the change in these 4 sums.
        
        # Old contributions from row i and col i:
        # R_i_old = sum_k a[i,k] b[pi, p[k]]
        # C_i_old = sum_k a[k,i] b[p[k], pi]
        # R_j_old = sum_k a[j,k] b[pj, p[k]]
        # C_j_old = sum_k a[k,j] b[p[k], pj]
        
        # New contributions:
        # R_i_new = sum_k a[i,k] b[pj, p_new[k]]  (p_new[i]=pj, p_new[j]=pi)
        # C_i_new = sum_k a[k,i] b[p_new[k], pj]
        # R_j_new = sum_k a[j,k] b[pi, p_new[k]]
        # C_j_new = sum_k a[k,j] b[p_new[k], pi]
        
        # We can compute these using dot products.
        
        # b[pi, p] is a vector. b[pj, p] is a vector.
        # b[p, pi] is a vector. b[p, pj] is a vector.
        
        v_bi = b[pi, p]
        v_bj = b[pj, p]
        v_pb_i = b[p, pi]
        v_pb_j = b[p, pj]
        
        # Old
        R_i_old = np.dot(a[i, :], v_bi)
        C_i_old = np.dot(a[:, i], v_pb_i)
        R_j_old = np.dot(a[j, :], v_bj)
        C_j_old = np.dot(a[:, j], v_pb_j)
        
        # New
        # For R_i_new: b[pj, p_new[k]]
        # p_new[k] is p[k] for k!=i,j.
        # p_new[i] = pj.
        # p_new[j] = pi.
        # So b[pj, p_new] is v_bj but with index i replaced by b[pj, pj] and index j replaced by b[pj, pi].
        # v_bj[i] is b[pj, p[i]] = b[pj, pi]. We need b[pj, pj].
        # v_bj[j] is b[pj, p[j]] = b[pj, pj]. We need b[pj, pi].
        
        v_bj_new = v_bj.copy()
        v_bj_new[i] = b[pj, pj]
        v_bj_new[j] = b[pj, pi]
        
        R_i_new = np.dot(a[i, :], v_bj_new)
        
        # For C_i_new: b[p_new[k], pj]
        # p_new[k] is p[k] for k!=i,j.
        # p_new[i] = pj.
        # p_new[j] = pi.
        # v_pb_j[k] is b[p[k], pj].
        # We need b[p_new[k], pj].
        # At k=i: b[pj, pj].
        # At k=j: b[pi, pj].
        
        v_pb_j_new = v_pb_j.copy()
        v_pb_j_new[i] = b[pj, pj]
        v_pb_j_new[j] = b[pi, pj]
        
        C_i_new = np.dot(a[:, i], v_pb_j_new)
        
        # For R_j_new: b[pi, p_new[k]]
        # v_bi[k] is b[pi, p[k]].
        # At k=i: b[pi, pj].
        # At k=j: b[pi, pi].
        
        v_bi_new = v_bi.copy()
        v_bi_new[i] = b[pi, pj]
        v_bi_new[j] = b[pi, pi]
        
        R_j_new = np.dot(a[j, :], v_bi_new)
        
        # For C_j_new: b[p_new[k], pi]
        # v_pb_i[k] is b[p[k], pi].
        # At k=i: b[pj, pi].
        # At k=j: b[pi, pi].
        
        v_pb_i_new = v_pb_i.copy()
        v_pb_i_new[i] = b[pj, pi]
        v_pb_i_new[j] = b[pi, pi]
        
        C_j_new = np.dot(a[:, j], v_pb_i_new)
        
        # The total cost change is the sum of these new terms minus old terms.
        # Note: The terms (i,i), (i,j), (j,i), (j,j) are included in these sums.
        # R_i includes k=i (a[i,i]b[pi,pi]) and k=j (a[i,j]b[pi,pj]).
        # C_i includes k=i (a[i,i]b[pi,pi]) and k=j (a[j,i]b[pj,pi]).
        # R_j includes k=i (a[j,i]b[pj,pi]) and k=j (a[j,j]b[pj,pj]).
        # C_j includes k=i (a[i,j]b[pi,pj]) and k=j (a[j,j]b[pj,pj]).
        
        # In the new sums:
        # R_i_new includes k=i (a[i,i]b[pj,pj]) and k=j (a[i,j]b[pj,pi]).
        # C_i_new includes k=i (a[i,i]b[pj,pj]) and k=j (a[j,i]b[pi,pj]).
        # R_j_new includes k=i (a[j,i]b[pi,pj]) and k=j (a[j,j]b[pi,pi]).
        # C_j_new includes k=i (a[i,j]b[pi,pj]) and k=j (a[j,j]b[pi,pi]).
        
        # So we just sum them.
        
        old_cost_part = R_i_old + C_i_old + R_j_old + C_j_old
        new_cost_part = R_i_new + C_i_new + R_j_new + C_j_new
        
        return new_cost_part - old_cost_part

    # 1. Initialization: Hungarian-based rank structure
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    proxy_cost = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Iterated Local Search (ILS)
    # This is more effective than SA for QAP often.
    
    def local_search(p, current_cost, max_iter=10000):
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() - start_time < time_limit:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    delta = delta_swap(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    # Run initial local search
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    # ILS Loop
    while time.time() - start_time < time_limit:
        # Perturbation
        # Random 2-3 swaps
        num_swaps = random.randint(2, 5)
        p = best_p.copy()
        current_cost = best_cost
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            delta = delta_swap(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
