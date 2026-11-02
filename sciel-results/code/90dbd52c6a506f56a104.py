"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Precompute support sets for each dimension of the tensor
    # For index (i, j, k) in tensor T:
    # U index is i*m + j
    # V index is j*p + k
    # W index is k*n + i
    
    # We want to find a set of columns (u_idx, v_idx, w_idx) such that
    # for every (i, j, k) in {0..n-1} x {0..m-1} x {0..p-1},
    # there exists a column c such that:
    # U[i*m+j, c] != 0
    # V[j*p+k, c] != 0
    # W[k*n+i, c] != 0
    
    # This is a hitting set problem.
    # Elements: all (i, j, k)
    # Sets: for each column c, the set of (i, j, k) covered by non-zeros in U[:, c], V[:, c], W[:, c].
    # A column c covers (i, j, k) if U[i*m+j, c] != 0 AND V[j*p+k, c] != 0 AND W[k*n+i, c] != 0.
    
    # To make this tractable, we can restrict the search space for U, V, W entries.
    # Given the "Strassen" hint, maybe we can look for structured solutions?
    # Or just use the greedy hitting set approach with random restarts on the *values*.
    
    # Let's try a different approach:
    # 1. Generate random sparse U, V, W matrices with entries from {-1, 0, 1}.
    # 2. Check if the tensor is covered.
    # 3. If not, use local search to flip bits to cover more.
    
    # Actually, the power method approach in the current code tries to find *values* that sum to 1.
    # The hitting set approach finds *supports* that cover the tensor.
    # Once we have supports, we need to find values.
    # If supports form a "perfect" cover (each element covered exactly once), values are 1.
    # If not, we need to solve a linear system or use least squares.
    
    # Let's stick to the power method but improve the initialization and search.
    # The key insight from "Strassen recursion" might be that we should try to find
    # decompositions that are not just random, but have some structure.
    # However, for 2x5x6, the structure is not obvious.
    
    # Let's try to improve the local search by using a better initialization.
    # Instead of random integers, initialize with the schoolbook solution and then
    # try to merge columns?
    
    # Another idea: Use the fact that the tensor is sparse.
    # T has 60 non-zeros.
    # We want to approximate T with sum of r rank-1 tensors.
    
    # Let's try to search for ranks from 55 down to 40.
    # For each rank, use random restarts with the power method.
    
    # Improvement: 
    # 1. Use a time budget per rank.
    # 2. If a rank is not found quickly, move to the next lower rank? 
    #    No, lower rank is harder. 
    #    So we should spend more time on higher ranks first?
    #    Actually, if we find rank 55, we should try 54.
    #    If we fail to find 55, we should try 54.
    #    So the loop structure is correct.
    
    # Let's optimize the power method part.
    # The current code computes `current_sum` and `residual` in every iteration.
    # This is O(r * n*m*m*p*p*n) which is expensive.
    # We can maintain the sum incrementally.
    
    for r in range(59, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        # Allocate time based on rank?
        # Let's just try a fixed number of restarts, but break early if time is up.
        num_restarts = 100
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W
            # Random small integers
            U = np.random.randint(-1, 2, (n*m, r)).astype(float)
            V = np.random.randint(-1, 2, (m*p, r)).astype(float)
            W = np.random.randint(-1, 2, (p*n, r)).astype(float)
            
            # Precompute initial sum
            current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
            
            converged = False
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                    
                residual = T - current_sum
                
                # Check convergence
                if np.linalg.norm(residual) < 1e-9:
                    converged = True
                    break
                
                # Select a random column to update
                c = np.random.randint(r)
                
                # Compute residual excluding column c
                # sum_without_c = current_sum - U[:, c] o V[:, c] o W[:, c]
                # target_for_c = T - sum_without_c = residual + U[:, c] o V[:, c] o W[:, c]
                col_sum = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                target_for_c = residual + col_sum
                
                # Power method to find best rank-1 approx to target_for_c
                u = U[:, c].copy()
                v = V[:, c].copy()
                w = W[:, c].copy()
                
                # If vectors are zero, reinitialize
                if np.linalg.norm(u) < 1e-6: u = np.random.randn(n*m)
                if np.linalg.norm(v) < 1e-6: v = np.random.randn(m*p)
                if np.linalg.norm(w) < 1e-6: w = np.random.randn(p*n)
                
                for _ in range(10): # Few power iterations
                    # u_new = target_for_c contracted with v, w
                    u_new = np.einsum('ijk,j,k->i', target_for_c, v, w)
                    v_new = np.einsum('ijk,i,k->j', target_for_c, u, w)
                    w_new = np.einsum('ijk,i,j->k', target_for_c, u, v)
                    
                    norm_u = np.linalg.norm(u_new)
                    norm_v = np.linalg.norm(v_new)
                    norm_w = np.linalg.norm(w_new)
                    
                    if norm_u == 0 or norm_v == 0 or norm_w == 0:
                        break
                    
                    u_new /= norm_u
                    v_new /= norm_v
                    w_new /= norm_w
                    
                    u, v, w = u_new, v_new, w_new
                
                # Scale to match magnitude
                uo = np.einsum('i,j,k->ijk', u, v, w)
                denom = np.sum(uo * uo)
                if denom == 0:
                    continue
                s = np.sum(target_for_c * uo) / denom
                
                # Update column c
                U[:, c] = s * u
                V[:, c] = s * v
                W[:, c] = s * w
                
                # Update current_sum incrementally
                # current_sum = current_sum - old_col + new_col
                new_col = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                current_sum = current_sum - col_sum + new_col
                
            if converged:
                # We have a real-valued solution.
                # Try to round to half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    # Convert to lists
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    # Found a valid rank r. 
                    # We can break the restart loop and try lower rank.
                    break
        # If we found a rank, we continue to the next lower rank in the outer loop.
        # If we didn't find a rank, we also continue to the next lower rank.
        # Wait, if we didn't find rank 59, should we try 58?
        # Yes, 58 is easier than 59.
        # So the loop `for r in range(59, 39, -1)` is correct.

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
