import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Convert to numpy for faster operations
    U_s = np.array(U_school, dtype=float)
    V_s = np.array(V_school, dtype=float)
    W_s = np.array(W_school, dtype=float)
    
    # Target tensor T (all ones)
    T = np.ones((n, m, p), dtype=float)
    
    # Try to find a lower rank solution
    # We start from rank_school - 1 down to rank_school - 5
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = 140.0  # 140s limit
    
    for target_rank in range(rank_school - 1, max(rank_school - 10, 40), -1):
        if time.time() > deadline:
            break
            
        # Try random subsets of columns of size target_rank
        # Or just try the first target_rank columns
        # To improve chances, we can try a few random permutations
        
        for trial in range(5):
            if time.time() > deadline:
                break
                
            # Select target_rank columns
            if trial == 0:
                indices = list(range(target_rank))
            else:
                indices = np.random.choice(rank_school, target_rank, replace=False).tolist()
            
            U_sub = U_s[:, indices]
            V_sub = V_s[:, indices]
            W_sub = W_s[:, indices]
            
            # We want to find W' such that sum_r U_sub[:, :, r] * V_sub[:, :, r] * W'[:, :, r] = T
            # This is linear in W' if we fix U, V.
            # Equation: sum_r W'_{k,i,r} * (U_{i,j,r} * V_{j,k,r}) = T_{i,j,k}
            
            # Let's set up the linear system for W'
            # Variables: W'_{k,i,r} for k in 0..p-1, i in 0..n-1, r in 0..target_rank-1
            # Total variables: p * n * target_rank
            num_vars = p * n * target_rank
            
            # Matrix A (size 54 x num_vars)
            # Row index: (i, j, k) -> i*m*p + j*p + k
            # Col index: (k, i, r) -> k*n*target_rank + i*target_rank + r
            
            A = np.zeros((n * m * p, num_vars), dtype=float)
            b = np.zeros(n * m * p, dtype=float)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        b[row] = 1.0
                        for r in range(target_rank):
                            col = k * n * target_rank + i * target_rank + r
                            A[row, col] = U_sub[i * m + j, r] * V_sub[j * p + k, r]
            
            # Solve A W' = b
            try:
                W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is close to integer/half-integer
            W_flat = W_sol.flatten()
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Check error
            W_rounded_matrix = W_rounded.reshape(p, n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_sub[i * m + j, r] * V_sub[j * p + k, r] * W_rounded_matrix[k, i, r]
                        error += abs(val - 1.0)
            
            if error < 1e-9:
                # Found a valid solution
                U_new = U_sub.tolist()
                V_new = V_sub.tolist()
                W_new = W_rounded_matrix.tolist()
                
                # Ensure integer/half-integer format
                # The problem allows integer/half-integer
                return {"U": U_new, "V": V_new, "W": W_new}
            
            # If not exact, we can't use it directly.
            # But maybe we can adjust?
            # For now, just continue searching.
            
    # If no lower rank found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
