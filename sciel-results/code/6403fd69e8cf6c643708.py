"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit (150s kill)
    
    # Helper: Calculate exact cost
    def calc_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    # Helper: Calculate delta for swapping facilities i and j in permutation p
    # This is O(n) if we precompute some things, but here we do O(n) per swap
    # For n=256, O(n) is 256 operations, which is very fast in numpy.
    def swap_delta(p, i, j):
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        
        # Cost change = sum_k (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        # Note: The standard formula for symmetric QAP is:
        # delta = 2 * ( sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]]) )
        #       + (a[i,i] - a[j,j]) * (b[lj, lj] - b[li, li])
        #       + (a[i,j] - a[j,i]) * (b[lj, li] - b[li, lj])  -- this term is 0 if symmetric a and b?
        # Let's use the robust O(n) formula derived from expanding the cost function.
        
        # Let C(p) = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Swapping i and j changes p[i] and p[j].
        # Terms involving i or j:
        # 1. a[i,i] b[p[i], p[i]] -> a[i,i] b[lj, lj]
        # 2. a[j,j] b[p[j], p[j]] -> a[j,j] b[li, li]
        # 3. a[i,j] b[p[i], p[j]] -> a[i,j] b[lj, li]
        # 4. a[j,i] b[p[j], p[i]] -> a[j,i] b[li, lj]
        # 5. sum_{k != i,j} a[i,k] b[p[i], p[k]] -> sum_{k != i,j} a[i,k] b[lj, p[k]]
        # 6. sum_{k != i,j} a[j,k] b[p[j], p[k]] -> sum_{k != i,j} a[j,k] b[li, p[k]]
        # 7. sum_{k != i,j} a[k,i] b[p[k], p[i]] -> sum_{k != i,j} a[k,i] b[p[k], lj]
        # 8. sum_{k != i,j} a[k,j] b[p[k], p[j]] -> sum_{k != i,j} a[k,j] b[p[k], li]
        
        # If a and b are symmetric:
        # a[i,j] = a[j,i], b[lj,li] = b[li,lj]
        # Term 3 + 4 = 2 * a[i,j] * b[lj, li]
        # Old Term 3 + 4 = 2 * a[i,j] * b[li, lj]
        # Change = 2 * a[i,j] * (b[lj, li] - b[li, lj]) = 0 if b symmetric.
        
        # Terms 5+7 combine for a symmetric a:
        # sum_{k != i,j} (a[i,k] + a[k,i]) * (b[lj, p[k]] - b[li, p[k]]) 
        # = 2 * sum_{k != i,j} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
        
        # Terms 6+8 combine for a symmetric a:
        # sum_{k != i,j} (a[j,k] + a[k,j]) * (b[li, p[k]] - b[lj, p[k]])
        # = 2 * sum_{k != i,j} a[j,k] * (b[li, p[k]] - b[lj, p[k]])
        # = -2 * sum_{k != i,j} a[j,k] * (b[lj, p[k]] - b[li, p[k]])
        
        # Total off-diagonal change:
        # 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        
        # Diagonal change:
        # a[i,i] * (b[lj, lj] - b[li, li]) + a[j,j] * (b[li, li] - b[lj, lj])
        # = (a[i,i] - a[j,j]) * (b[lj, lj] - b[li, li])
        
        # So, delta = (a[i,i] - a[j,j]) * (b[lj, lj] - b[li, li]) 
        #            + 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, p[k]] - b[li, p[k]])
        
        # Vectorized computation for the sum part:
        # Let diff_a = a[i, :] - a[j, :]
        # Let diff_b = b[lj, p] - b[li, p]
        # Sum over all k is dot(diff_a, diff_b)
        # We need to exclude k=i and k=j.
        
        diff_a = a[i] - a[j]
        diff_b = b[lj, p] - b[li, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract term for k=i:
        # diff_a[i] * diff_b[i] = (a[i,i] - a[j,i]) * (b[lj, p[i]] - b[li, p[i]])
        # p[i] = li
        # = (a[i,i] - a[j,i]) * (b[lj, li] - b[li, li])
        
        # Subtract term for k=j:
        # diff_a[j] * diff_b[j] = (a[i,j] - a[j,j]) * (b[lj, p[j]] - b[li, p[j]])
        # p[j] = lj
        # = (a[i,j] - a[j,j]) * (b[lj, lj] - b[li, lj])
        
        term_i = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li])
        term_j = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj])
        
        off_diag_sum = total_dot - term_i - term_j
        
        diag_delta = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
        
        return int(diag_delta + 2 * off_diag_sum)

    # Strategy: Continuous Graph Embedding via Sinkhorn-Knopp
    # 1. Initialize soft assignment matrix P (n x n)
    # 2. Iteratively refine P using gradient descent on the relaxed QAP cost
    #    Cost(P) = sum_{i,j,k,l} a[i,j] b[k,l] P[i,k] P[j,l]
    #    This is equivalent to minimizing ||A^{1/2} P B^{1/2} P^T A^{1/2}||_F^2 or similar
    #    But simpler: we can use the fact that if P is a permutation matrix, cost is sum a[i,j] b[p[i],p[j]]
    #    The relaxation allows P to be a doubly stochastic matrix.
    #    We minimize J(P) = sum_{i,j} a[i,j] (P^T B P)[i,j]
    #    Gradient wrt P: G = 2 * A P B^T + 2 * A^T P B (if symmetric A,B)
    #    Actually, let's derive it properly.
    #    Let C = P^T B P. Then J = sum_{i,j} A[i,j] C[i,j] = trace(A^T C) = trace(A^T P^T B P) = trace(P A^T P^T B)
    #    If A and B are symmetric:
    #    J = sum_{i,j,k,l} A[i,j] P[i,k] B[k,l] P[j,l]
    #    dJ/dP[i,k] = sum_{j,l} A[i,j] B[k,l] P[j,l] + sum_{j,l} A[j,i] B[l,k] P[j,l]
    #               = (A P B^T)[i,k] + (A^T P B)[i,k]
    #    If A and B are symmetric, this is 2 * (A P B)[i,k].
    #    So we perform gradient descent on P, then project back to the Birkhoff polytope using Sinkhorn-Knopp.
    
    # Initialize P
    # Start with a random doubly stochastic matrix or the one from Hungarian on row sums
    # Let's use the Hungarian initialization as a hard assignment, then relax it slightly
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P_init = np.zeros((n, n), dtype=np.float64)
    P_init[row_ind, col_ind] = 1.0
    
    # Add noise to break symmetry and allow exploration
    P = P_init + 0.1 * np.random.rand(n, n)
    P = P / P.sum() # Normalize
    
    # Sinkhorn-Knopp projection to doubly stochastic
    def sinkhorn(P, n_iter=50, eps=1e-6):
        P = P.copy()
        for _ in range(n_iter):
            # Row normalization
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P = P / row_sums
            # Column normalization
            col_sums = P.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            P = P / col_sums
        return P

    P = sinkhorn(P)
    
    # Gradient Descent on P
    # We want to minimize J(P) = sum_{i,j} A[i,j] (P^T B P)[i,j]
    # Gradient G = 2 * A P B (assuming symmetric A, B)
    # Update rule: P <- P - eta * G
    # Then project to Birkhoff polytope.
    
    # Precompute A and B as float64 for stability
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # Adaptive learning rate
    eta = 0.1
    best_P = P.copy()
    best_cost = calc_cost(np.argmax(P, axis=1))
    
    # We will run multiple iterations of Sinkhorn + Gradient Step
    # To allow for diversification, we can run multiple restarts with different initializations
    # But first, let's see how the single continuous search performs.
    
    max_iterations = 2000
    no_improve_count = 0
    patience = 50
    
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        # Compute Gradient
        # G = 2 * A P B
        G = 2.0 * (A @ P @ B)
        
        # Update P
        P_new = P - eta * G
        
        # Project to Birkhoff Polytope
        P_new = sinkhorn(P_new, n_iter=20)
        
        # Check for improvement in the relaxed objective or the extracted permutation
        # Extract permutation
        perm = np.argmax(P_new, axis=1)
        # Ensure it's a valid permutation (might have conflicts)
        # If conflicts, use Hungarian on the current P_new to get a valid permutation
        # This is a good way to "repair" errors in the continuous embedding
        row_ind, col_ind = linear_sum_assignment(-P_new) # Maximize P_new
        perm_hung = np.empty(n, dtype=np.int64)
        perm_hung[row_ind] = col_ind
        
        cost_hung = calc_cost(perm_hung)
        
        if cost_hung < best_cost:
            best_cost = cost_hung
            best_P = P_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        P = P_new
        
        # Adaptive learning rate
        if no_improve_count > patience:
            eta *= 0.5
            no_improve_count = 0
            if eta < 1e-4:
                break
                
    # Final extraction
    row_ind, col_ind = linear_sum_assignment(-best_P)
    best_perm = np.empty(n, dtype=np.int64)
    best_perm[row_ind] = col_ind
    
    # Local Search on the best permutation found
    current_p = best_perm.copy()
    current_cost = best_cost
    
    # Run local search to polish the result
    max_passes = 20
    for pass_num in range(max_passes):
        if time.time() > deadline - 2.0:
            break
        improved = False
        start_i = random.randint(0, n - 1)
        for k in range(n):
            i = (start_i + k) % n
            if time.time() > deadline - 2.0:
                break
            u = current_p[i]
            start_j = random.randint(0, n - 1)
            for m in range(n):
                j = (start_j + m) % n
                if i == j:
                    continue
                if time.time() > deadline - 2.0:
                    break
                delta = swap_delta(current_p, i, j)
                if delta < 0:
                    v = current_p[j]
                    current_p[i], current_p[j] = v, u
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_perm = current_p.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    # If local search improved, use it
    if current_cost < best_cost:
        best_perm = current_p
        best_cost = current_cost

    # Final validation
    if sorted(best_perm.tolist()) != list(range(n)):
        return list(range(n))
        
    return best_perm.tolist()
# EVOLVE-BLOCK-END
