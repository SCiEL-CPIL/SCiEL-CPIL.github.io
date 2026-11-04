import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    facility_order = np.argsort(-flow_sums, kind="stable")
    location_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    # 2. Calculate Initial Cost
    b_perm = b[np.ix_(p, p)]
    current_cost = np.sum(a * b_perm)
    best_cost = current_cost
    best_p = p.copy()
    
    # 3. Tabu Search Setup
    start_time = time.time()
    time_limit = 115.0
    
    tabu_length = 10
    tabu_queue = [] # List of (i, j) in order of insertion
    tabu_set = set() # For O(1) lookup
    
    # Optimized Delta Cost Calculation
    def delta_cost(i, j, p):
        li = p[i]
        lj = p[j]
        
        b_li_P = b[li, p]
        b_lj_P = b[lj, p]
        b_P_li = b[p, li]
        b_P_lj = b[p, lj]
        
        term1 = np.dot(a[i, :], b_lj_P - b_li_P)
        term2 = np.dot(a[:, i], b_P_lj - b_P_li)
        term3 = np.dot(a[j, :], b_li_P - b_lj_P)
        term4 = np.dot(a[:, j], b_P_li - b_P_lj)
        
        delta = term1 + term2 + term3 + term4
        
        # Correction for diagonals (assumed 0 in QAPLIB, but let's be robust)
        # The vectorized sum includes terms for k=i and k=j which are incorrect for the swap logic.
        # Garbage at k=i: -a[j,i]*b[lj,li] - a[i,j]*b[li,lj]
        # Garbage at k=j: -a[i,j]*b[li,lj] - a[j,i]*b[lj,li]
        # Correct delta for pair (i,j): a[i,j](b[lj,li]-b[li,lj]) + a[j,i](b[li,lj]-b[lj,li])
        
        g_i = -a[j,i]*b[lj,li] - a[i,j]*b[li,lj]
        g_j = -a[i,j]*b[li,lj] - a[j,i]*b[lj,li]
        delta_pair = a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
        
        delta = delta - g_i - g_j + delta_pair
        
        return delta

    while time.time() - start_time < time_limit:
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Evaluate all pairs
        for i in range(n):
            for j in range(i + 1, n):
                move_key = (i, j)
                is_tabu = move_key in tabu_set
                
                dc = delta_cost(i, j, p)
                
                # Aspiration: Accept tabu move if it improves best_cost
                if is_tabu:
                    new_cost = current_cost + dc
                    if new_cost < best_cost:
                        if dc < best_delta:
                            best_delta = dc
                            best_i = i
                            best_j = j
                else:
                    if dc < best_delta:
                        best_delta = dc
                        best_i = i
                        best_j = j
                        
        if best_i == -1:
            # No move found (should not happen unless all are tabu and none improve best)
            # Clear tabu list to allow exploration
            tabu_queue.clear()
            tabu_set.clear()
            continue
            
        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu Structures
        tabu_queue.append((best_i, best_j))
        tabu_set.add((best_i, best_j))
        
        if len(tabu_queue) > tabu_length:
            old_move = tabu_queue.pop(0)
            tabu_set.remove(old_move)
            
    return best_p.tolist()
