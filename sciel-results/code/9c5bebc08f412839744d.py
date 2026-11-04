import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s limit
    
    # 1. Initialization: Rank Matching
    # Facilities with high total flow -> Locations with low total distance
    row_sum_a = a.sum(axis=1)
    row_sum_b = b.sum(axis=1)
    
    fac_order = np.argsort(-row_sum_a, kind='stable')
    loc_order = np.argsort(row_sum_b, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    def calc_cost(p):
        # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        return np.sum(a * b[p, :][:, p])
        
    best_p = p.copy()
    best_cost = calc_cost(p)
    
    def local_search(p):
        current_cost = calc_cost(p)
        # M[i, l] = sum_j a[i, j] b[p[j], l]
        # M = a @ b[p, :]
        M = a @ b[p, :]
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            
            # Compute Delta matrix
            # Delta[i, j] = M[i, p[j]] - M[i, p[i]] + M[j, p[i]] - M[j, p[j]]
            M_at_P = M[:, p] # M_at_P[i, j] = M[i, p[j]]
            diag_M_at_P = np.diag(M_at_P)
            
            # X[i, j] = M[i, p[j]] - M[i, p[i]]
            X = M_at_P - diag_M_at_P[:, None]
            
            # Delta[i, j] = X[i, j] + X[j, i]
            Delta = X + X.T
            
            # Mask diagonal (self-swaps)
            np.fill_diagonal(Delta, np.inf)
            
            # Find best swap
            min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
            min_delta = Delta[min_idx]
            
            if min_delta >= 0:
                improved = False
                break
            
            i, j = min_idx
            if i == j:
                # Should not happen due to inf diagonal, but safe
                improved = False
                break
                
            u = p[i]
            v = p[j]
            
            # Perform swap
            p[i], p[j] = p[j], p[i]
            
            # Update M
            # M_new[k, l] = M_old[k, l] + (a[k, j] - a[k, i]) * (b[u, l] - b[v, l])
            diff_a = a[:, j] - a[:, i]
            diff_b = b[u, :] - b[v, :]
            M += np.outer(diff_a, diff_b)
            
            current_cost += min_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        return p, current_cost

    # Iterated Local Search
    p_current = p.copy()
    
    while time.time() - start_time < time_limit:
        p_local, cost_local = local_search(p_current)
        
        if cost_local < best_cost:
            best_cost = cost_local
            best_p = p_local.copy()
        
        # Check time before perturbation
        if time.time() - start_time > time_limit - 5:
            break
            
        # Perturb: 5 random swaps
        for _ in range(5):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_local[i], p_local[j] = p_local[j], p_local[i]
        
        p_current = p_local

    return best_p.tolist()
