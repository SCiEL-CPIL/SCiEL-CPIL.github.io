# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute constants
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Check symmetry for potential optimizations (though we use general formulas for safety)
    # QAPLIB tai256c is symmetric, but let's be robust.
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)

    # --- Helper: compute cost ---
    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    # --- Helper: 2-opt delta (vectorized) ---
    def calc_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        if u == v:
            return 0

        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]

        diff_b_row = b[v, :] - b[u, :]
        diff_b_col = b[:, v] - b[:, u]

        C = diff_b_row[perm]
        D = diff_b_col[perm]

        full_sum_1 = np.dot(diff_a_row, C)
        full_sum_2 = np.dot(diff_a_col, D)

        corr_i_1 = diff_a_row[i] * C[i]
        corr_i_2 = diff_a_col[i] * D[i]

        corr_j_1 = diff_a_row[j] * C[j]
        corr_j_2 = diff_a_col[j] * D[j]

        sum_off_diag = (full_sum_1 - corr_i_1 - corr_j_1) + (full_sum_2 - corr_i_2 - corr_j_2)

        direct_delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return int(sum_off_diag + direct_delta)

    # --- Helper: 2-opt local search polish ---
    def two_opt_polish(p, cost, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = calc_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Strategy 1: Frank-Wolfe (Conditional Gradient) on Birkhoff Polytope ---
    # This provides a global perspective by relaxing the discrete constraint.
    def frank_wolfe_search(start_p, max_time):
        # Initialize X as a permutation matrix
        X = np.zeros((n, n), dtype=np.float64)
        X[np.arange(n), start_p] = 1.0
        
        best_p = start_p.copy()
        best_cost = calc_cost_full(start_p)
        
        t = 0
        while time.time() < max_time:
            # Compute Gradient G of the QAP objective w.r.t X
            # F(X) = sum_{i,j,k,l} a[i,k] b[j,l] X[i,j] X[k,l]
            # G[i,j] = sum_{k,l} (a[i,k] b[j,l] X[k,l] + a[k,j] b[i,l] X[k,l])
            
            if a_sym and b_sym:
                # Optimization for symmetric case
                # G = 2 * (A @ X @ B)
                G = 2.0 * (a @ X @ b)
            else:
                G = (a @ X @ b.T) + (a.T @ X @ b)
            
            # Linear Assignment Problem (LAP) to find vertex P
            # Minimize <G, P>
            row_ind, col_ind = linear_sum_assignment(G)
            
            # P is the permutation matrix corresponding to col_ind
            # We don't need to explicitly construct P for the update if we just use the permutation
            # X_new = (1 - gamma) X + gamma P
            # But P is sparse. 
            # However, X is dense. 
            # Let's update X.
            
            gamma = 2.0 / (t + 2.0)
            
            # Construct P implicitly in the update?
            # X = (1-gamma)*X + gamma*P
            # P[i, col_ind[i]] = 1
            
            # To avoid creating a full dense P matrix every time (though n=256 is small),
            # we can do:
            # X *= (1 - gamma)
            # Then add gamma to the positions of P
            X *= (1.0 - gamma)
            X[np.arange(n), col_ind] += gamma
            
            # Extract permutation from X?
            # The current vertex P (col_ind) is a candidate.
            # Also, we could try to project X to the nearest permutation matrix, 
            # but the LAP solution col_ind is already a vertex of the polytope 
            # that minimizes the linear approximation. 
            # Using col_ind directly as the candidate permutation is standard in CG for QAP.
            
            p_candidate = col_ind.copy()
            cost_candidate = calc_cost_full(p_candidate)
            
            if cost_candidate < best_cost:
                best_cost = cost_candidate
                best_p = p_candidate.copy()
            
            t += 1
            
            # Early stop if no improvement for a long time? 
            # The linear_sum_assignment is O(n^3). For n=256, it's fast.
            # We rely on the time limit.

        return best_p, best_cost

    # --- Initialization: Rank Matching ---
    # Sort facilities by flow sum (heavy to light)
    # Sort locations by distance sum (central to peripheral)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order
    
    # Initial Polish
    p_best = p_init.copy()
    cost_best = calc_cost_full(p_best)
    p_best, cost_best = two_opt_polish(p_best, cost_best, time.time() + 5.0)

    # --- Main Loop: Frank-Wolfe + Polish + Simulated Annealing ---
    
    # Phase 1: Frank-Wolfe to find a good basin
    fw_deadline = time.time() + 40.0
    p_fw, cost_fw = frank_wolfe_search(p_best, fw_deadline)
    
    if cost_fw < cost_best:
        cost_best = cost_fw
        p_best = p_fw.copy()
        
    # Polish the FW result
    p_polish = p_best.copy()
    cost_polish = cost_best
    p_polish, cost_polish = two_opt_polish(p_polish, cost_polish, time.time() + 10.0)
    
    if cost_polish < cost_best:
        cost_best = cost_polish
        p_best = p_polish.copy()

    # Phase 2: Simulated Annealing to escape local optima
    # Use the remaining time.
    sa_deadline = deadline - 5.0
    
    if time.time() < sa_deadline:
        p_sa = p_best.copy()
        cost_sa = cost_best
        
        # Tune SA parameters
        T = 1000.0
        T_min = 1.0
        alpha = 0.9995
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < sa_deadline:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            delta = calc_delta(i, j, p_sa)
            
            # Accept if improving, or with probability exp(-delta/T) if worsening
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                
                if cost_sa < cost_best:
                    cost_best = cost_sa
                    p_best = p_sa.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > max_no_improve:
                    # Restart temperature or increase it slightly to escape
                    T = max(T * 1.5, 50.0)
                    no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T * 1.5, 50.0)
                    no_improve = 0
                    
            T = max(T * alpha, T_min)

    # Phase 3: Final Polish
    p_final = p_best.copy()
    cost_final = cost_best
    p_final, cost_final = two_opt_polish(p_final, cost_final, deadline)
    
    if cost_final < cost_best:
        cost_best = cost_final
        p_best = p_final.copy()

    return p_best.tolist()

# EVOLVE-BLOCK-END
