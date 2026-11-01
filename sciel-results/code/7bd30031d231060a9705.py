import time
import numpy as np
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Function to check if a decomposition is valid (exact integer/half-integer)
    def is_valid(U, V, W, rank):
        # Check if entries are close to multiples of 0.5
        # We round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if the rounded matrices form an exact decomposition
        # Tensor T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r] should be 1
        # We can check a sample of entries or all entries
        # 64 entries is small, check all
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    # Try to find a lower-rank decomposition
    # We iterate ranks from 32 to 63
    for target_rank in range(32, 64):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_inits = 20
        for init in range(max_inits):
            if time.time() > deadline:
                break
            
            np.random.seed(init * 1000 + target_rank)
            
            # Initialize randomly
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # 3-way Alternating Minimization
            for it in range(150):
                if time.time() > deadline:
                    break
                
                # 1. Update U: Fix V, W
                # For each (i,j), solve A_ij * u_ij = 1
                # A_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        # Least squares
                        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j, :] = u_sol
                        
                # 2. Update V: Fix U, W
                # For each (j,k), solve A_jk * v_jk = 1
                # A_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k, :] = v_sol
                        
                # 3. Update W: Fix U, V
                # For each (k,i), solve A_ki * w_ki = 1
                # A_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w_sol
                        
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residual += (val - 1)**2
                
                if residual < 1e-8:
                    # Try to round and verify
                    if is_valid(U, V, W, target_rank):
                        # Round to nearest 0.5 for final output
                        U_final = np.round(U * 2) / 2
                        V_final = np.round(V * 2) / 2
                        W_final = np.round(W * 2) / 2
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                    else:
                        # If rounding fails, we might still be close. 
                        # But the requirement is exact. 
                        # We continue searching or break to try next init.
                        pass
                        
                if time.time() > deadline:
                    break

    # Return the best we found (schoolbook)
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
