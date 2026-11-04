import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Hungarian Algorithm Initialization
    # Linearized score: W[i,j] = sum_k a[i,k] * b[j,k]
    # We want to minimize the total linearized cost
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Calculate initial cost
    B_perm = b[:, p]
    S = a @ B_perm
    current_cost = int(np.sum(S[np.arange(n), p]))
    best_cost = current_cost
    best_p = p.copy()
    
    # 2. Simulated Annealing with 2-opt
    # Estimate initial temperature
    deltas = []
    for _ in range(50):
        i, k = np.random.choice(n, 2, replace=False)
        u, v = p[i], p[k]
        # Compute delta for swap i<->k
        # delta = sum_{j!=i,k} [a[i,j]*b[v,p[j]] + a[j,i]*b[p[j],v] + a[k,j]*b[u,p[j]] + a[j,k]*b[p[j],u]
        #                  - a[i,j]*b[u,p[j]] - a[j,i]*b[p[j],u] - a[k,j]*b[v,p[j]] - a[j,k]*b[p[j],v]]
        #       + [a[i,k]*b[v,u] + a[k,i]*b[u,v] - a[i,k]*b[u,v] - a[k,i]*b[v,u]]
        
        # Vectorized:
        # Let r = p (current assignment)
        # For each j: term_j = (a[i,j]-a[k,j])*(b[v,r[j]]-b[u,r[j]]) + (a[j,i]-a[j,k])*(b[r[j],v]-b[r[j],u])
        # Sum over j != i,k, then add the i-k cross term
        
        r = p
        # Full sum over all j
        diff_a_i = a[i, :] - a[k, :]
        diff_a_j = a[:, i] - a[:, k]
        b_v_r = b[v, r]
        b_u_r = b[u, r]
        b_r_v = b[r, v]
        b_r_u = b[r, u]
        
        terms = diff_a_i * (b_v_r - b_u_r) + diff_a_j * (b_r_v - b_r_u)
        total_sum = np.sum(terms)
        
        # Subtract j=i and j=k terms (they should not be included)
        # j=i: r[i]=u
        # term_i = diff_a_i[i] * (b[v,u]-b[u,u]) + diff_a_j[i] * (b[u,v]-b[u,u])
        term_i = (a[i,i]-a[k,i]) * (b[v,u]-b[u,u]) + (a[i,i]-a[i,k]) * (b[u,v]-b[u,u])
        # j=k: r[k]=v
        term_k = (a[i,k]-a[k,k]) * (b[v,v]-b[u,v]) + (a[k,i]-a[k,k]) * (b[v,v]-b[v,u])
        
        sum_part = total_sum - term_i - term_k
        
        # Cross term between i and k
        cross = a[i,k]*b[v,u] + a[k,i]*b[u,v] - a[i,k]*b[u,v] - a[k,i]*b[v,u]
        
        delta = sum_part + cross
        deltas.append(abs(delta))
    
    T = np.mean(deltas) * 2.0 if deltas else 1000.0
    T_min = 1e-6
    alpha = 0.999
    
    start_time = time.time()
    time_limit = 105.0
    
    while time.time() - start_time < time_limit:
        i, k = np.random.choice(n, 2, replace=False)
        u, v = p[i], p[k]
        
        r = p
        diff_a_i = a[i, :] - a[k, :]
        diff_a_j = a[:, i] - a[:, k]
        b_v_r = b[v, r]
        b_u_r = b[u, r]
        b_r_v = b[r, v]
        b_r_u = b[r, u]
        
        terms = diff_a_i * (b_v_r - b_u_r) + diff_a_j * (b_r_v - b_r_u)
        total_sum = np.sum(terms)
        
        term_i = (a[i,i]-a[k,i]) * (b[v,u]-b[u,u]) + (a[i,i]-a[i,k]) * (b[u,v]-b[u,u])
        term_k = (a[i,k]-a[k,k]) * (b[v,v]-b[u,v]) + (a[k,i]-a[k,k]) * (b[v,v]-b[v,u])
        
        sum_part = total_sum - term_i - term_k
        cross = a[i,k]*b[v,u] + a[k,i]*b[u,v] - a[i,k]*b[u,v] - a[k,i]*b[v,u]
        delta = sum_part + cross
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            p[i], p[k] = v, u
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # 3. 2-opt local search on best solution
    p = best_p.copy()
    current_cost = best_cost
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for k in range(i+1, n):
                u, v = p[i], p[k]
                r = p
                diff_a_i = a[i, :] - a[k, :]
                diff_a_j = a[:, i] - a[:, k]
                b_v_r = b[v, r]
                b_u_r = b[u, r]
                b_r_v = b[r, v]
                b_r_u = b[r, u]
                
                terms = diff_a_i * (b_v_r - b_u_r) + diff_a_j * (b_r_v - b_r_u)
                total_sum = np.sum(terms)
                
                term_i = (a[i,i]-a[k,i]) * (b[v,u]-b[u,u]) + (a[i,i]-a[i,k]) * (b[u,v]-b[u,u])
                term_k = (a[i,k]-a[k,k]) * (b[v,v]-b[u,v]) + (a[k,i]-a[k,k]) * (b[v,v]-b[v,u])
                
                sum_part = total_sum - term_i - term_k
                cross = a[i,k]*b[v,u] + a[k,i]*b[u,v] - a[i,k]*b[u,v] - a[k,i]*b[v,u]
                delta = sum_part + cross
                
                if delta < 0:
                    p[i], p[k] = v, u
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()
