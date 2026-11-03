# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute constants
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

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

    # --- Helper: Frank-Wolfe (Conditional Gradient) on Birkhoff Polytope ---
    def frank_wolfe_init(max_time):
        # Start with a random permutation or rank matching
        p_init = np.random.permutation(n)
        
        # We maintain a doubly stochastic matrix X.
        # To save memory and time, we don't explicitly store X if it's sparse, 
        # but for n=256, a dense (256,256) float array is small (~500KB).
        # However, updating X via rank-1 updates (outer products) is efficient.
        
        # Initialize X as a permutation matrix for p_init
        X = np.zeros((n, n), dtype=np.float64)
        X[np.arange(n), p_init] = 1.0
        
        # Gradient of the QAP objective w.r.t X is:
        # G[i, j] = sum_k (a[i, k] * X[k, j] + X[i, k] * b[k, j]) 
        # Wait, the objective is sum_{i,j,k,l} a[i,k] b[j,l] X[i,j] X[k,l]
        # Gradient G[i,j] = sum_{k,l} (a[i,k] b[j,l] X[k,l] + a[k,j] b[i,l] X[k,l]) ? No.
        # Let's use the standard linearization for QAP on Birkhoff polytope.
        # F(X) = <A, X> <B, X> ? No.
        # F(X) = sum_{i,j} a[i,j] * (sum_k X[i,k] b[k, ?] ... )
        
        # The standard linearization for a fixed permutation P (or matrix X) is:
        # The gradient G for the linear term in the CG method is:
        # G[i, j] = sum_k (a[i, k] * X[k, j] + X[i, k] * b[k, j]) 
        # Actually, for QAP, the gradient of F(X) = sum_{i,j,k,l} a[i,k]b[j,l]X[i,j]X[k,l] is:
        # G[i,j] = sum_{k,l} (a[i,k]b[j,l]X[k,l] + a[k,j]b[i,l]X[k,l])
        
        # Let's precompute M = X @ B (or similar) to speed up gradient.
        # G = A @ X @ B^T + A^T @ X @ B ?
        # Let's check dimensions.
        # Term 1: sum_k a[i,k] (sum_l b[j,l] X[k,l]) = sum_k a[i,k] (B X^T)[j,k] ? 
        # Let Y = X @ B^T. Then Y[k,l] = sum_j X[k,j] B[j,l] = sum_j X[k,j] B[l,j] (if B sym? No).
        # Actually, let's just compute G directly using matrix mults.
        # G = A @ X @ B + A.T @ X @ B.T ? 
        # Let's verify with indices.
        # (A @ X @ B)[i, j] = sum_{k,l} A[i,k] X[k,l] B[l,j]
        # (A.T @ X @ B.T)[i, j] = sum_{k,l} A[k,i] X[k,l] B[j,l]
        # Sum: sum_{k,l} X[k,l] (A[i,k]B[l,j] + A[k,i]B[j,l])
        # This matches the gradient formula if we identify indices correctly.
        # In QAP: F = sum_{i,j,k,l} a[i,k] b[j,l] x[i,j] x[k,l]
        # dF/dx[i,j] = sum_{k,l} (a[i,k] b[j,l] x[k,l] + a[k,j] b[i,l] x[k,l])
        # Note: b[j,l] is element (j,l) of B. In matrix mult A @ X @ B, we have B[l,j].
        # So we need B^T in the second position?
        # Let's use: G = A @ X @ B.T + A.T @ X @ B
        # (A @ X @ B.T)[i,j] = sum_{k,l} A[i,k] X[k,l] B.T[l,j] = sum_{k,l} A[i,k] X[k,l] B[j,l]
        # (A.T @ X @ B)[i,j] = sum_{k,l} A.T[i,k] X[k,l] B[l,j] = sum_{k,l} A[k,i] X[k,l] B[l,j]
        # Sum: sum_{k,l} X[k,l] (A[i,k]B[j,l] + A[k,i]B[l,j])
        # This matches dF/dx[i,j] if we swap indices in the second term?
        # dF/dx[i,j] has term a[k,j] b[i,l]. My derivation has A[k,i] B[l,j].
        # If A and B are symmetric (common in QAP), A[k,i]=A[i,k] and B[l,j]=B[j,l].
        # tai256c: Are A and B symmetric?
        # QAPLIB instances typically have symmetric flow and distance matrices.
        # If they are symmetric, G = 2 * (A @ X @ B).
        # Let's assume symmetry for speed, but check.
        # If not symmetric, we use the full formula.
        
        if np.allclose(a, a.T) and np.allclose(b, b.T):
            sym = True
        else:
            sym = False

        X = np.zeros((n, n), dtype=np.float64)
        # Start from a good heuristic permutation
        heavy_order = np.argsort(-a_row_sums, kind='stable')
        central_order = np.argsort(b_row_sums, kind='stable')
        p_init = np.empty(n, dtype=np.int64)
        p_init[heavy_order] = central_order
        X[np.arange(n), p_init] = 1.0
        
        best_p = p_init.copy()
        best_cost = calc_cost_full(p_init)
        
        gamma = 1.0 / 2.0  # Step size parameter
        t = 0
        while time.time() < max_time:
            # Compute Gradient
            if sym:
                # G = 2 * (A @ X @ B)
                G = 2.0 * (a @ X @ b)
            else:
                G = (a @ X @ b.T) + (a.T @ X @ b)
            
            # Linear Assignment Problem (LAP) to find vertex P of Birkhoff polytope
            # Minimize sum G[i,j] X[i,j]
            row_ind, col_ind = linear_sum_assignment(G)
            P = np.zeros((n, n), dtype=np.float64)
            P[row_ind, col_ind] = 1.0
            
            # Update X: X_new = (1 - gamma) X + gamma P
            # gamma = 2 / (t+2)
            gamma = 2.0 / (t + 2.0)
            X = (1.0 - gamma) * X + gamma * P
            
            # Project back to Birkhoff polytope?
            # The convex combination of two points in the Birkhoff polytope is in the Birkhoff polytope.
            # So X remains doubly stochastic.
            
            # Extract permutation from X?
            # X is not a permutation matrix. We need a permutation to evaluate cost.
            # We can use the LAP solution P as the current candidate permutation.
            # Or we can use a more sophisticated projection.
            # For now, use P as the candidate.
            p_candidate = col_ind.copy()
            cost_candidate = calc_cost_full(p_candidate)
            
            if cost_candidate < best_cost:
                best_cost = cost_candidate
                best_p = p_candidate.copy()
            
            t += 1
            
            # Check convergence or time
            if time.time() > max_time:
                break

        return best_p, best_cost

    # --- Strategy ---
    # 1. Quick rank matching init
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # 2. Polish it
    p, cost = two_opt_polish(p, calc_cost_full(p), time.time() + 10.0)
    best_p = p.copy()
    best_cost = cost

    # 3. Run Frank-Wolfe for a while
    # Allocate ~50s for Frank-Wolfe
    fw_deadline = time.time() + 50.0
    p_fw, cost_fw = frank_wolfe_init(fw_deadline)
    
    if cost_fw < best_cost:
        best_cost = cost_fw
        best_p = p_fw.copy()
    else:
        # If FW didn't improve, stick with previous
        pass

    # 4. Polish the best result found so far
    p_polish = best_p.copy()
    cost_polish = best_cost
    p_polish, cost_polish = two_opt_polish(p_polish, cost_polish, time.time() + 20.0)
    
    if cost_polish < best_cost:
        best_cost = cost_polish
        best_p = p_polish.copy()

    # 5. Simulated Annealing on the best solution to escape local optima
    # Use remaining time
    if time.time() < deadline - 5.0:
        p_sa = best_p.copy()
        cost_sa = best_cost
        
        T = 100.0
        T_min = 1.0
        alpha = 0.999
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5.0:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            delta = calc_delta(i, j, p_sa)
            
            if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > max_no_improve:
                    T = max(T * 1.5, 50.0)
                    no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    T = max(T * 1.5, 50.0)
                    no_improve = 0
                    
            T = max(T * alpha, T_min)

    # 6. Final Polish
    p_final = best_p.copy()
    cost_final = best_cost
    p_final, cost_final = two_opt_polish(p_final, cost_final, deadline)
    
    if cost_final < best_cost:
        best_cost = cost_final
        best_p = p_final.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
