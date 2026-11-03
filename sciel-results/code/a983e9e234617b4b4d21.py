# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    # Frank-Wolfe initialization on Birkhoff polytope
    # Start with a doubly stochastic matrix (uniform)
    X = np.ones((n, n), dtype=np.float64) / n
    
    # Precompute a_row_sums and b_row_sums for the linearized cost
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    fw_steps = 30
    fw_time_budget = 20.0  # seconds for Frank-Wolfe
    
    for step in range(fw_steps):
        if time.time() > start_time + fw_time_budget:
            break
        
        # Compute gradient of QAP objective w.r.t. X
        # Objective: f(X) = sum_{i,j,k,l} a[i,j] * X[i,k] * b[k,l] * X[j,l]
        # Gradient: grad[i,k] = sum_{j,l} a[i,j] * b[k,l] * X[j,l] + sum_{j,l} a[j,i] * b[l,k] * X[j,l]
        # More precisely: grad[i,k] = sum_j a[i,j] * (B @ X)[j,k] + sum_j a[j,i] * (X @ B)[j,k]
        
        # grad = (A @ (B @ X)) + (A.T @ (X @ B)).T
        BX = b @ X
        XB = X @ b
        grad = (a @ BX) + (a.T @ XB).T
        
        # Frank-Wolfe step: find S = argmin_{S in Birkhoff} <grad, S>
        # This is a linear assignment problem
        row_ind, col_ind = linear_sum_assignment(grad)
        S = np.zeros((n, n), dtype=np.float64)
        S[row_ind, col_ind] = 1.0
        
        # Compute step size (line search)
        # Try a few step sizes
        best_alpha = 0.0
        best_val = np.sum(grad * X)
        
        for alpha in [0.5, 0.25, 0.1, 0.05, 0.01]:
            X_trial = (1 - alpha) * X + alpha * S
            # Compute objective value
            obj = np.sum(a * (X_trial.T @ b @ X_trial))
            if obj < best_val:
                best_val = obj
                best_alpha = alpha
        
        if best_alpha == 0.0:
            break
        X = (1 - best_alpha) * X + best_alpha * S
    
    # Extract permutation from the final doubly stochastic matrix
    row_ind, col_ind = linear_sum_assignment(-X)
    p_fw = col_ind.copy()
    
    # Also compute rank matching as a fallback
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_order] = central_order
    
    # Choose the better initial solution
    cost_fw = calc_cost_full(p_fw)
    cost_rank = calc_cost_full(p_rank)
    
    if cost_fw <= cost_rank:
        p = p_fw.copy()
        current_cost = cost_fw
    else:
        p = p_rank.copy()
        current_cost = cost_rank
    
    best_p = p.copy()
    best_cost = current_cost

    # Precompute parts for delta calculation speedup
    # We will use the vectorized delta formula derived in the thought process
    
    def calc_delta(i, j, perm):
        """
        Calculate the change in cost if we swap facilities i and j.
        i and j are indices in the facility array.
        perm[i] is the current location of facility i.
        perm[j] is the current location of facility j.
        """
        u = perm[i]
        v = perm[j]
        
        # If u == v, delta is 0 (should not happen in permutation)
        if u == v:
            return 0
            
        # A_diff_row = a[i, :] - a[j, :]
        # A_diff_col = a[:, i] - a[:, j]
        # B_diff_row = b[v, :] - b[u, :]
        # B_diff_col = b[:, v] - b[:, u]
        
        # We need:
        # Sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, perm[k]] - b[u, perm[k]]) 
        # + Sum_{k != i, j} (a[k,i] - a[k,j]) * (b[perm[k], v] - b[perm[k], u])
        # + (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        
        # Vectorized computation:
        # Let C[k] = b[v, perm[k]] - b[u, perm[k]]
        # Let D[k] = b[perm[k], v] - b[perm[k], u]
        
        # C = (b[v, :] - b[u, :])[perm]
        # D = (b[:, v] - b[:, u])[perm]
        
        # Term1_full = dot(a[i, :] - a[j, :], C)
        # Term2_full = dot(a[:, i] - a[:, j], D)
        
        # Subtract k=i and k=j terms from full sums
        # Then add direct interaction delta
        
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, :] - b[u, :]
        diff_b_col = b[:, v] - b[:, u]
        
        # C vector: indexed by facility k, value is diff in b for location perm[k]
        # C[k] = diff_b_row[perm[k]]
        C = diff_b_row[perm]
        # D vector: indexed by facility k, value is diff in b for location perm[k] (col v - col u)
        # D[k] = diff_b_col[perm[k]]
        D = diff_b_col[perm]
        
        full_sum_1 = np.dot(diff_a_row, C)
        full_sum_2 = np.dot(diff_a_col, D)
        
        # Corrections for k=i
        # C[i] = diff_b_row[perm[i]] = diff_b_row[u] = b[v,u] - b[u,u]
        # D[i] = diff_b_col[perm[i]] = diff_b_col[u] = b[u,v] - b[u,u]
        corr_i_1 = diff_a_row[i] * C[i]
        corr_i_2 = diff_a_col[i] * D[i]
        
        # Corrections for k=j
        # C[j] = diff_b_row[perm[j]] = diff_b_row[v] = b[v,v] - b[u,v]
        # D[j] = diff_b_col[perm[j]] = diff_b_col[v] = b[v,v] - b[v,u]
        corr_j_1 = diff_a_row[j] * C[j]
        corr_j_2 = diff_a_col[j] * D[j]
        
        sum_off_diag = (full_sum_1 - corr_i_1 - corr_j_1) + (full_sum_2 - corr_i_2 - corr_j_2)
        
        # Direct interaction delta
        # Old: a[i,j]*b[u,v] + a[j,i]*b[v,u]
        # New: a[i,j]*b[v,u] + a[j,i]*b[u,v]
        # Delta: a[i,j]*(b[v,u] - b[u,v]) + a[j,i]*(b[u,v] - b[v,u])
        #       = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        
        direct_delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        return int(sum_off_diag + direct_delta)

    # Simulated Annealing Parameters
    T = 500.0
    T_min = 1.0
    alpha = 0.9995
    no_improve_count = 0
    max_no_improve = 10000
    
    # Main SA Loop
    while time.time() < deadline - 10.0:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = calc_delta(i, j, p)
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Reheat if stuck?
            if no_improve_count > max_no_improve:
                T = max(T * 1.5, 100.0)
                no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                T = max(T * 1.5, 100.0)
                no_improve_count = 0

        T = max(T * alpha, T_min)

    # Final Polish: Strict 2-Opt Local Search
    # Use the remaining time to polish the best solution found
    # We polish on a copy of the best solution to ensure we don't lose it
    p_polish = best_p.copy()
    cost_polish = best_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    break
                delta = calc_delta(i, j, p_polish)
                if delta < 0:
                    p_polish[i], p_polish[j] = p_polish[j], p_polish[i]
                    cost_polish += delta
                    if cost_polish < best_cost:
                        best_cost = cost_polish
                        best_p = p_polish.copy()
                    improved = True
                    break
            if improved:
                break

    return best_p.tolist()

# EVOLVE-BLOCK-END
