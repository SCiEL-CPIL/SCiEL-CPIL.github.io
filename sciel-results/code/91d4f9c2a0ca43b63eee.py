import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target ranks to try, starting from low
    # Known rank for 2x3x3 is 16
    target_ranks = [16, 17]
    
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    # Time limit guard
    start_time = time.time()
    time_limit = 110  # 10s margin
    
    for target_rank in target_ranks:
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a decomposition of this rank
        found = False
        max_iters = 2000 # Adjust based on time
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization of U and V
            # Use small integers/half-integers to bias towards sparse/simple solutions?
            # Or just random floats?
            # Let's try random floats in [-2, 2]
            U_rand = np.random.uniform(-2, 2, (n * m, target_rank))
            V_rand = np.random.uniform(-2, 2, (m * p, target_rank))
            
            # Construct linear system for W
            # Equations: sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k
            # Number of equations: n*m*p
            # Number of unknowns: p*n*target_rank
            num_eqs = n * m * p
            num_unk = p * n * target_rank
            
            L = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            # Fill L
            # Index mapping:
            # eq_idx: (i, j, k) -> i*m*p + j*p + k
            # unk_idx: (k, r) -> k*n + r  (Wait, W is p x n? No, W is p*n x rank? 
            # Task says W shape (p*n, rank).
            # Index of W is k*n + i.
            # So unknowns are W[k*n+i, r].
            # Let's map unk_idx to (k, i, r) -> (k*n+i)*target_rank + r
            
            # Let's re-verify the tensor definition.
            # C[i,k] = sum_j A[i,j] B[j,k]
            # T[i,j,k] = 1
            # Decomposition: sum_r U[i,r] V[j,r] W[k,r] ?
            # No, the third index of the tensor corresponds to the output index.
            # Output is C[i,k].
            # So the tensor is T[i, j, (i,k)]? No.
            # The tensor is in U_A \otimes U_B \otimes U_C.
            # U_A dim n*m = 6.
            # U_B dim m*p = 9.
            # U_C dim p*n = 6. (Output C is 2x3, flattened as k*n+i? 
            # Task says: "output factor follows the official trace convention W[k*n+i,r]".
            # So W index is k*n+i.
            
            # So the equation is:
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            
            # Let's fill L accordingly.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # For this eq, we need W[k*n+i, r]
                        # Coefficient is U[i*m+j, r] * V[j*p+k, r]
                        for r in range(target_rank):
                            unk_idx = (k * n + i) * target_rank + r
                            L[eq_idx, unk_idx] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
            
            # Solve L W_flat = b
            try:
                W_flat, residuals, rank_L, sv = np.linalg.lstsq(L, b, rcond=None)
            except:
                continue
                
            if residuals is not None and residuals[0] > 1e-6:
                continue
                
            W_sol = W_flat.reshape(p * n, target_rank)
            
            # Check if U, V, W are close to half-integers
            # We can try to round them and check the error.
            U_rounded = np.round(U_rand * 2) / 2
            V_rounded = np.round(V_rand * 2) / 2
            W_rounded = np.round(W_sol * 2) / 2
            
            # Calculate error with rounded matrices
            # sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i]
            err = 0
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                        max_err = max(max_err, abs(val - 1))
                        
            if max_err < 1e-9:
                # Found a valid decomposition!
                found = True
                best_rank = target_rank
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                break
        
        if found:
            break
            
    return best_result
