"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization using Linear Assignment Problem on a heuristic matrix
    # Heuristic: Match high flow sum to low distance sum
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Matrix for LAP: C[i, j] = a_row_sums[i] * b_row_sums[j]
    # Minimizing this matches high a_row_sum with low b_row_sum
    C = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # 2. Sinkhorn-Gradient Descent Phase
    # Initialize P as permutation matrix
    P = np.zeros((n, n), dtype=np.float64)
    P[np.arange(n), p] = 1.0
    
    # Gradient Descent with Sinkhorn Projection
    # Objective: tr(A P B^T P^T)
    # Gradient w.r.t P: A P B + B P A^T (assuming symmetric A, B)
    # We assume A and B are symmetric as per standard QAP instances like tai256c
    
    alpha = 0.01 # Learning rate
    sinkhorn_iters = 10
    gd_iters = 100
    
    best_p = p.copy()
    best_cost = np.inf
    
    def compute_cost(p):
        # Efficient cost calculation
        # Cost = sum_i (a[i, :] @ b[p[i], p])
        # b[p[i], p] is row p[i] of b, indexed by p
        # Let B_perm = b[:, p]
        # Then b[p[i], p] is B_perm[p[i], :]? No.
        # b[p[i], p[j]] is element at row p[i], col p[j].
        # Let B_perm = b[:, p]. Then B_perm[k, j] = b[k, p[j]].
        # So b[p[i], p[j]] = B_perm[p[i], j].
        # Cost = sum_i sum_j a[i, j] * B_perm[p[i], j]
        # Cost = sum_i (a[i, :] @ B_perm[p[i], :])
        B_perm = b[:, p]
        cost = 0
        for i in range(n):
            cost += np.dot(a[i], B_perm[p[i]])
        return cost

    current_cost = compute_cost(p)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    for _ in range(gd_iters):
        # Compute Gradient
        # G = A @ P @ B + B @ P @ A
        G = a @ P @ b + b @ P @ a
        
        # Gradient Step
        P = P - alpha * G
        
        # Clip to ensure positivity
        P = np.clip(P, 1e-6, None)
        
        # Sinkhorn-Knopp Iterations to restore doubly stochastic property
        for _ in range(sinkhorn_iters):
            # Normalize rows
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / row_sums
            # Normalize cols
            col_sums = P.sum(axis=0, keepdims=True)
            P = P / col_sums
            
        # Extract permutation from P
        # Use LAP on -P to get the closest permutation
        row_ind, col_ind = linear_sum_assignment(-P)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = compute_cost(p_new)
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p = p_new.copy()
            # Reset P to the new permutation matrix to stay close to the manifold
            P = np.zeros((n, n), dtype=np.float64)
            P[np.arange(n), p] = 1.0
        else:
            # If no improvement, keep P but maybe decay alpha or just continue
            pass

    # 3. Local Search Phase (2-Swap)
    # Start from best_p found in Sinkhorn-GD
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute B_perm = b[:, p]
    B_perm = b[:, p]
    
    # Helper to update B_perm after swap i, j
    # Swapping p[i] and p[j] means swapping columns i and j in B_perm
    
    max_ls_passes = 20
    improved = True
    pass_count = 0
    
    while improved and pass_count < max_ls_passes:
        improved = False
        pass_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = p[i], p[j]
                
                # Calculate delta cost for swapping i and j
                # Old terms involving i or j:
                # Sum_{k != i,j} (a[i,k] b[pi, p[k]] + a[k,i] b[p[k], pi] + a[j,k] b[pj, p[k]] + a[k,j] b[p[k], pj])
                # + a[i,i] b[pi, pi] + a[i,j] b[pi, pj] + a[j,i] b[pj, pi] + a[j,j] b[pj, pj]
                
                # New terms:
                # Sum_{k != i,j} (a[i,k] b[pj, p[k]] + a[k,i] b[p[k], pj] + a[j,k] b[pi, p[k]] + a[k,j] b[p[k], pi])
                # + a[i,i] b[pj, pj] + a[i,j] b[pj, pi] + a[j,i] b[pi, pj] + a[j,j] b[pi, pi]
                
                # Delta = New - Old
                
                # Efficient calculation:
                # Let B_perm_col_k = B_perm[:, k] which is b[:, p[k]]
                # b[x, p[k]] = B_perm[x, k]
                
                # Sum over k != i, j:
                # Term i: sum_{k != i,j} a[i,k] * (B_perm[pj, k] - B_perm[pi, k])
                # Term i^T: sum_{k != i,j} a[k,i] * (B_perm[k, pj] - B_perm[k, pi])
                # Term j: sum_{k != i,j} a[j,k] * (B_perm[pi, k] - B_perm[pj, k])
                # Term j^T: sum_{k != i,j} a[k,j] * (B_perm[k, pi] - B_perm[k, pj])
                
                # Note: B_perm[k, x] = b[k, p[x]]
                
                # We can compute these sums using dot products and subtracting the i, j terms.
                
                # S1 = sum_{k} a[i,k] * (B_perm[pj, k] - B_perm[pi, k])
                # S2 = sum_{k} a[k,i] * (B_perm[k, pj] - B_perm[k, pi])
                # S3 = sum_{k} a[j,k] * (B_perm[pi, k] - B_perm[pj, k])
                # S4 = sum_{k} a[k,j] * (B_perm[k, pi] - B_perm[k, pj])
                
                # S1 = a[i, :] @ (B_perm[pj, :] - B_perm[pi, :])
                # S2 = a[:, i] @ (B_perm[:, pj] - B_perm[:, pi])
                # S3 = a[j, :] @ (B_perm[pi, :] - B_perm[pj, :])
                # S4 = a[:, j] @ (B_perm[:, pi] - B_perm[:, pj])
                
                diff_row_pi_pj = B_perm[pi, :] - B_perm[pj, :]
                diff_col_pi_pj = B_perm[:, pi] - B_perm[:, pj]
                
                S1 = np.dot(a[i, :], diff_row_pi_pj)
                S2 = np.dot(a[:, i], diff_col_pi_pj)
                S3 = np.dot(a[j, :], -diff_row_pi_pj)
                S4 = np.dot(a[:, j], -diff_col_pi_pj)
                
                # These sums include k=i and k=j. We need to remove them and add the correct diagonal/cross terms.
                
                # Remove k=i from S1: a[i,i] * (B_perm[pj, i] - B_perm[pi, i])
                # B_perm[pj, i] = b[pj, p[i]] = b[pj, pi]
                # B_perm[pi, i] = b[pi, p[i]] = b[pi, pi]
                rem_S1_i = a[i, i] * (b[pj, pi] - b[pi, pi])
                
                # Remove k=j from S1: a[i,j] * (B_perm[pj, j] - B_perm[pi, j])
                # B_perm[pj, j] = b[pj, p[j]] = b[pj, pj]
                # B_perm[pi, j] = b[pi, p[j]] = b[pi, pj]
                rem_S1_j = a[i, j] * (b[pj, pj] - b[pi, pj])
                
                # Remove k=i from S2: a[i,i] * (B_perm[i, pj] - B_perm[i, pi])
                # B_perm[i, pj] = b[p[i], pj] = b[pi, pj]
                # B_perm[i, pi] = b[p[i], pi] = b[pi, pi]
                rem_S2_i = a[i, i] * (b[pi, pj] - b[pi, pi])
                
                # Remove k=j from S2: a[j,i] * (B_perm[j, pj] - B_perm[j, pi])
                # B_perm[j, pj] = b[p[j], pj] = b[pj, pj]
                # B_perm[j, pi] = b[p[j], pi] = b[pj, pi]
                rem_S2_j = a[j, i] * (b[pj, pj] - b[pj, pi])
                
                # Remove k=i from S3: a[j,i] * (B_perm[pi, i] - B_perm[pj, i])
                # B_perm[pi, i] = b[pi, pi]
                # B_perm[pj, i] = b[pj, pi]
                rem_S3_i = a[j, i] * (b[pi, pi] - b[pj, pi])
                
                # Remove k=j from S3: a[j,j] * (B_perm[pi, j] - B_perm[pj, j])
                # B_perm[pi, j] = b[pi, pj]
                # B_perm[pj, j] = b[pj, pj]
                rem_S3_j = a[j, j] * (b[pi, pj] - b[pj, pj])
                
                # Remove k=i from S4: a[i,j] * (B_perm[i, pi] - B_perm[i, pj])
                # B_perm[i, pi] = b[pi, pi]
                # B_perm[i, pj] = b[pi, pj]
                rem_S4_i = a[i, j] * (b[pi, pi] - b[pi, pj])
                
                # Remove k=j from S4: a[j,j] * (B_perm[j, pi] - B_perm[j, pj])
                # B_perm[j, pi] = b[pj, pi]
                # B_perm[j, pj] = b[pj, pj]
                rem_S4_j = a[j, j] * (b[pj, pi] - b[pj, pj])
                
                sum_S1 = S1 - rem_S1_i - rem_S1_j
                sum_S2 = S2 - rem_S2_i - rem_S2_j
                sum_S3 = S3 - rem_S3_i - rem_S3_j
                sum_S4 = S4 - rem_S4_i - rem_S4_j
                
                # Diagonal and cross terms for i, j
                # Old Diagonal/Cross:
                old_diag = a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj]
                old_cross = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
                
                # New Diagonal/Cross:
                new_diag = a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]
                new_cross = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
                
                delta = (sum_S1 + sum_S2 + sum_S3 + sum_S4) + (new_diag - old_diag) + (new_cross - old_cross)
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = pj, pi
                    current_cost += delta
                    improved = True
                    # Update B_perm: swap columns i and j
                    B_perm[:, [i, j]] = B_perm[:, [j, i]]
    
    return p.tolist()
# EVOLVE-BLOCK-END
