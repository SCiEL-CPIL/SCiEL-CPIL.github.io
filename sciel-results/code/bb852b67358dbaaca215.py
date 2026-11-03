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

    # Precompute a_row_sums and b_row_sums for the linearized cost
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Quick Frank-Wolfe initialization with fewer steps and tighter time budget
    X = np.ones((n, n), dtype=np.float64) / n
    fw_steps = 15
    fw_time_budget = 8.0  # seconds for Frank-Wolfe

    for step in range(fw_steps):
        if time.time() > start_time + fw_time_budget:
            break

        # Compute gradient: grad = (A @ (B @ X)) + (A.T @ (X @ B)).T
        BX = b @ X
        XB = X @ b
        grad = (a @ BX) + (a.T @ XB).T

        # Frank-Wolfe step: find S = argmin_{S in Birkhoff} <grad, S>
        row_ind, col_ind = linear_sum_assignment(grad)
        S = np.zeros((n, n), dtype=np.float64)
        S[row_ind, col_ind] = 1.0

        # Use fixed step size (2/(t+2)) for simplicity and speed
        alpha = 2.0 / (step + 2)
        X = (1 - alpha) * X + alpha * S

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
        
        # Precompute differences
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, :] - b[u, :]
        diff_b_col = b[:, v] - b[:, u]
        
        # C[k] = diff_b_row[perm[k]]
        # D[k] = diff_b_col[perm[k]]
        C = diff_b_row[perm]
        D = diff_b_col[perm]
        
        full_sum_1 = np.dot(diff_a_row, C)
        full_sum_2 = np.dot(diff_a_col, D)
        
        # Corrections for k=i and k=j
        corr_i_1 = diff_a_row[i] * C[i]
        corr_i_2 = diff_a_col[i] * D[i]
        corr_j_1 = diff_a_row[j] * C[j]
        corr_j_2 = diff_a_col[j] * D[j]
        
        sum_off_diag = (full_sum_1 - corr_i_1 - corr_j_1) + (full_sum_2 - corr_i_2 - corr_j_2)
        
        # Direct interaction delta
        direct_delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        return int(sum_off_diag + direct_delta)

    # Simulated Annealing Parameters
    T = 500.0
    T_min = 1.0
    alpha = 0.9995
    no_improve_count = 0
    max_no_improve = 15000
    
    # Main SA Loop
    while time.time() < deadline - 15.0:
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
                
            # Reheat if stuck
            if no_improve_count > max_no_improve:
                T = max(T * 1.5, 100.0)
                no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                T = max(T * 1.5, 100.0)
                no_improve_count = 0

        T = max(T * alpha, T_min)

    # Final Polish: Steepest Descent 2-Opt Local Search
    # Use the remaining time to polish the best solution found
    p_polish = best_p.copy()
    cost_polish = best_cost
    
    polish_deadline = deadline - 2.0
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > polish_deadline:
                break
            for j in range(i + 1, n):
                if time.time() > polish_deadline:
                    break
                delta = calc_delta(i, j, p_polish)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            p_polish[best_i], p_polish[best_j] = p_polish[best_j], p_polish[best_i]
            cost_polish += best_delta
            if cost_polish < best_cost:
                best_cost = cost_polish
                best_p = p_polish.copy()
            improved = True

    return best_p.tolist()

# EVOLVE-BLOCK-END
