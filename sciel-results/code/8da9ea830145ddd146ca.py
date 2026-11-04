import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Greedy Initialization
    # Assign facilities one by one to the location that minimizes incremental cost
    p = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    
    # Order of assignment: Sort facilities by sum of flows in A (largest first)
    # This is a common heuristic to place high-flow facilities early
    a_sums = a.sum(axis=1)
    order = np.argsort(-a_sums)
    
    # To speed up greedy, we can just iterate.
    # Cost of assigning i to j given current p:
    # sum_{k: p[k]!=-1} a[i,k] * b[j, p[k]] + a[k,i] * b[p[k], j]
    
    # We maintain a matrix C[i, j] = current incremental cost to place i at j?
    # Or just compute on the fly. n=150, n^3 is small.
    
    for idx in order:
        best_loc = -1
        best_cost = np.inf
        for j in range(n):
            if p[j] != -1: # Location j is taken? No, p maps facility->location.
                # We need to track available locations.
                pass
        
    # Better: Track available locations as a set or boolean array.
    available = np.ones(n, dtype=bool)
    p = np.full(n, -1, dtype=np.int64)
    
    # Precompute B for fast access? b is already there.
    
    for idx in order:
        best_loc = -1
        best_cost = np.inf
        for j in range(n):
            if not available[j]:
                continue
            
            # Calculate incremental cost
            # sum_{k: p[k] != -1} (a[idx, k] * b[j, p[k]] + a[k, idx] * b[p[k], j])
            
            # Vectorized:
            # k_indices = np.where(p != -1)[0]
            # k_locs = p[k_indices]
            # cost = np.sum(a[idx, k_indices] * b[j, k_locs] + a[k_indices, idx] * b[k_locs, j])
            
            # This is O(n) per j. Total O(n^3).
            
            k_indices = np.where(p != -1)[0]
            if len(k_indices) == 0:
                cost = 0
            else:
                k_locs = p[k_indices]
                # a[idx, k_indices] is vector of len m
                # b[j, k_locs] is vector of len m
                # a[k_indices, idx] is vector of len m
                # b[k_locs, j] is vector of len m
                cost = np.sum(a[idx, k_indices] * b[j, k_locs] + a[k_indices, idx] * b[k_locs, j])
            
            if cost < best_cost:
                best_cost = cost
                best_loc = j
        
        p[idx] = best_loc
        available[best_loc] = False
        
    # Calculate initial cost and S
    # S[i, v] = sum_j a[i, j] * b[v, p[j]]
    B_perm = b[:, p]
    S = a @ B_perm
    current_cost = np.sum(S[np.arange(n), p])
    
    # 2. Simulated Annealing
    # Estimate initial temperature
    # Run a few random swaps to estimate delta magnitude
    T = 1000.0 # Placeholder
    deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        u, v = p[i], p[k]
        # Calculate delta
        V = b[:, v] - b[:, u]
        D = a[:, i] - a[:, k]
        sum_all = np.dot(D, V[p])
        term_i = (a[i, i] - a[i, k]) * (b[u, v] - b[u, u])
        term_k = (a[k, i] - a[k, k]) * (b[v, v] - b[v, u])
        sum_part = sum_all - term_i - term_k
        
        delta_1 = S[i, v] + (a[i, i] - a[i, k]) * (b[v, v] - b[v, u]) - S[i, u]
        delta_2 = S[k, u] + (a[k, i] - a[k, k]) * (b[u, v] - b[u, u]) - S[k, v]
        delta = delta_1 + delta_2 + sum_part
        deltas.append(abs(delta))
    
    if deltas:
        T = np.mean(deltas) * 2.0
    else:
        T = 1000.0
        
    T_min = 1e-4
    alpha = 0.9995 # Cooling rate
    # Or use time-based cooling
    
    start_time = time.time()
    time_limit = 110.0 # 10s buffer
    
    # Precompute random indices? No, generate on fly.
    
    while time.time() - start_time < time_limit:
        # Pick random i, k
        i, k = np.random.choice(n, 2, replace=False)
        u, v = p[i], p[k]
        
        # Calculate delta
        V = b[:, v] - b[:, u]
        D = a[:, i] - a[:, k]
        
        # sum_{j} D[j] * V[p[j]]
        sum_all = np.dot(D, V[p])
        
        # Subtract self terms
        # term_i = D[i] * V[p[i]] = (a[i,i]-a[i,k]) * (b[v,u]-b[u,u]) ? No.
        # V[p[i]] = V[u] = b[u,v] - b[u,u]
        # D[i] = a[i,i] - a[i,k]
        term_i = (a[i, i] - a[i, k]) * (b[u, v] - b[u, u])
        
        # term_k = D[k] * V[p[k]] = (a[k,i]-a[k,k]) * (b[v,v]-b[v,u])
        # V[p[k]] = V[v] = b[v,v] - b[v,u]
        # D[k] = a[k,i] - a[k,k]
        term_k = (a[k, i] - a[k, k]) * (b[v, v] - b[v, u])
        
        sum_part = sum_all - term_i - term_k
        
        # Delta for i and k themselves
        # S[i, v] - S[i, u] + S[k, u] - S[k, v] + adjustments for the swap effect on S[i,v] and S[k,u]
        # The S matrix update is S += outer(D, V)
        # So S'[i, v] = S[i, v] + D[i] * V[v]
        # S'[k, u] = S[k, u] + D[k] * V[u]
        
        delta_1 = (S[i, v] + (a[i, i] - a[i, k]) * (b[v, v] - b[v, u])) - S[i, u]
        delta_2 = (S[k, u] + (a[k, i] - a[k, k]) * (b[u, v] - b[u, u])) - S[k, v]
        
        delta = delta_1 + delta_2 + sum_part
        
        # Acceptance
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            # Apply swap
            p[i], p[k] = v, u
            
            # Update S
            S += np.outer(D, V)
            
            current_cost += delta
            
            # Update best solution if needed? 
            # For QAP, we just return the current state at the end.
            # But we can keep track of best_cost and best_p.
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    return p.tolist()
