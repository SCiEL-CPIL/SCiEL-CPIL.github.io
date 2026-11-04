"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Evaluator kills at 120s. Use 115s deadline for safety.
    deadline = start_time + 115.0

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def local_search_2opt(p, best_cost, a, b, max_passes=50):
        """Perform 2-opt local search. Optimized for speed."""
        # p is a numpy array
        B_perm = b[p].copy()  # B_perm[i] = b[p[i]]
        
        # Precompute A rows and cols as views for speed
        # We will iterate through pairs.
        
        for _pass in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            # Iterate over all pairs (i, j)
            # To avoid O(n^3) overhead in Python loops, we try to be as efficient as possible.
            # For n=150, O(n^2) pairs is 11k pairs. 
            # Each pair calculation involves O(n) dot products.
            # 11k * 150 = 1.6M operations per pass. This is fast.
            
            for i in range(n):
                if time.time() > deadline:
                    return p, best_cost
                ai = a[i]
                a_i_col = a[:, i]
                for j in range(i + 1, n):
                    aj = a[j]
                    a_j_col = a[:, j]
                    
                    # Vectorized delta calculation
                    # diff_row[k] = b[p[j], k] - b[p[i], k]
                    diff_row = B_perm[j] - B_perm[i]
                    # diff_col[k] = b[k, p[j]] - b[k, p[i]]
                    diff_col = B_perm[:, j] - B_perm[:, i]
                    
                    # d1 = sum_{k} (a[i,k] - a[j,k]) * diff_row[k]
                    d1 = (ai - aj) @ diff_row
                    # d2 = sum_{k} (a[k,i] - a[k,j]) * diff_col[k]
                    d2 = (a_i_col - a_j_col) @ diff_col
                    
                    # Correct for k=i and k=j terms which are not part of the "other" sum
                    # The formula sum_{k != i,j} ... 
                    # d1 includes k=i and k=j.
                    # Term k=i in d1: (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
                    # Term k=j in d1: (a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
                    
                    # d2 includes k=i and k=j.
                    # Term k=i in d2: (a[i,i] - a[i,j]) * (b[p[i], p[j]] - b[p[i], p[i]])
                    # Term k=j in d2: (a[j,i] - a[j,j]) * (b[p[j], p[j]] - b[p[j], p[i]])
                    
                    # Let's use the full sum and subtract the specific terms, then add the correct diagonal/cross terms.
                    # Actually, the standard delta formula for QAP swap (i,j) is:
                    # Delta = sum_{k != i,j} (A[i,k] - A[j,k]) * (B[p[j], p[k]] - B[p[i], p[k]]) 
                    #       + (A[k,i] - A[k,j]) * (B[p[k], p[j]] - B[p[k], p[i]])
                    #       + A[i,i]*(B[p[j],p[j]] - B[p[i],p[i]]) 
                    #       + A[j,j]*(B[p[i],p[i]] - B[p[j],p[j]])
                    #       + A[i,j]*(B[p[j],p[i]] - B[p[i],p[j]]) 
                    #       + A[j,i]*(B[p[i],p[j]] - B[p[j],p[i]])
                    
                    # My d1/d2 calculation above computes the sum over ALL k.
                    # Let's verify the subtraction.
                    
                    # d1_full = sum_k (ai[k] - aj[k]) * (B_perm[j][k] - B_perm[i][k])
                    # We want sum_{k != i,j}.
                    # Subtract k=i: (ai[i] - aj[i]) * (B_perm[j][i] - B_perm[i][i])
                    # Subtract k=j: (ai[j] - aj[j]) * (B_perm[j][j] - B_perm[i][j])
                    
                    term_ki_d1 = (ai[i] - aj[i]) * (B_perm[j, i] - B_perm[i, i])
                    term_kj_d1 = (ai[j] - aj[j]) * (B_perm[j, j] - B_perm[i, j])
                    d1_corrected = d1 - term_ki_d1 - term_kj_d1
                    
                    # d2_full = sum_k (a[k,i] - a[k,j]) * (B_perm[k,j] - B_perm[k,i])
                    # Subtract k=i: (a[i,i] - a[i,j]) * (B_perm[i,j] - B_perm[i,i])
                    # Subtract k=j: (a[j,i] - a[j,j]) * (B_perm[j,j] - B_perm[j,i])
                    
                    term_ki_d2 = (a_i_col[i] - a_i_col[j]) * (B_perm[i, j] - B_perm[i, i])
                    term_kj_d2 = (a_j_col[i] - a_j_col[j]) * (B_perm[j, j] - B_perm[j, i])
                    d2_corrected = d2 - term_ki_d2 - term_kj_d2
                    
                    # Now add the explicit diagonal and cross terms for the swap
                    # The terms involving i and j directly in the objective function change.
                    # Old: A[i,i]B[p[i],p[i]] + A[j,j]B[p[j],p[j]] + A[i,j]B[p[i],p[j]] + A[j,i]B[p[j],p[i]]
                    # New: A[i,i]B[p[j],p[j]] + A[j,j]B[p[i],p[i]] + A[i,j]B[p[j],p[i]] + A[j,i]B[p[i],p[j]]
                    
                    delta_diag_cross = (
                        ai[i] * (B_perm[j, j] - B_perm[i, i]) +
                        aj[j] * (B_perm[i, i] - B_perm[j, j]) +
                        ai[j] * (B_perm[j, i] - B_perm[i, j]) +
                        aj[i] * (B_perm[i, j] - B_perm[j, i])
                    )
                    
                    delta = d1_corrected + d2_corrected + delta_diag_cross
                    
                    if delta < 0:
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        B_perm[i], B_perm[j] = B_perm[j], B_perm[i]
                        best_cost += delta
                        improved = True
                        # First improvement strategy: break inner loop? 
                        # Or continue to find best? First improvement is faster.
                        # Let's stick to first improvement to allow more passes.
                        break 
                if improved:
                    # If we swapped, we might want to restart the outer loop or continue?
                    # Continuing is fine, but indices change. 
                    # For simplicity and correctness, if we swap, we break out and restart the pass.
                    break
            
            if not improved:
                break
                
        return p, best_cost

    def spectral_initialize(a, b, k=10):
        """Fast spectral initialization."""
        def double_center(M):
            n = len(M)
            J = np.eye(n) - np.ones((n, n)) / n
            return J @ M @ J

        C_A = double_center(a.astype(float))
        C_B = double_center(b.astype(float))
        
        # Use eigvalsh for symmetric matrices (faster)
        eigvals_A, eigvecs_A = np.linalg.eigh(C_A)
        eigvals_B, eigvecs_B = np.linalg.eigh(C_B)
        
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        eigvals_A = eigvals_A[idx_A]
        eigvecs_A = eigvecs_A[:, idx_A]
        
        eigvals_B = eigvals_B[idx_B]
        eigvecs_B = eigvecs_B[:, idx_B]
        
        X_A = eigvecs_A[:, :k] * np.sqrt(np.maximum(eigvals_A[:k], 0))
        X_B = eigvecs_B[:, :k] * np.sqrt(np.maximum(eigvals_B[:k], 0))
        
        M = X_A.T @ X_B
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        X_A_rot = X_A @ R
        
        norm_A = np.sum(X_A_rot**2, axis=1)
        norm_B = np.sum(X_B**2, axis=1)
        dists = norm_A[:, None] + norm_B[None, :] - 2 * (X_A_rot @ X_B.T)
        
        row_ind, col_ind = linear_sum_assignment(dists)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    # 1. Initialization
    p_init = spectral_initialize(a, b, k=10)
    best_p = p_init.copy()
    best_cost = compute_cost_fast(p_init)
    
    # 2. Local Search
    best_p, best_cost = local_search_2opt(best_p, best_cost, a, b)
    
    # 3. Random Restarts
    rng = np.random.RandomState(42)
    while time.time() < deadline - 2.0:
        p_rand = rng.permutation(n)
        p_rand, cost_rand = local_search_2opt(p_rand, compute_cost_fast(p_rand), a, b, max_passes=20)
        if cost_rand < best_cost:
            best_p = p_rand.copy()
            best_cost = cost_rand
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
