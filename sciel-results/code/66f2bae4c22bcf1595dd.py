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
    
    # Try to find a lower-rank decomposition
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try ranks from 32 up to 63
    for target_rank in range(32, 64):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_iters = 50
        for init in range(max_iters):
            if time.time() > deadline:
                break
            
            np.random.seed(init * 1000 + target_rank)
            
            # Initialize randomly
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Fix V, solve for U and W
                # For each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # This is bilinear in U and W for fixed V
                
                # Let's define: for each r, B_r[i,j,k] = V[j,k,r]
                # Then sum_r U[i,j,r]*B_r[i,j,k]*W[k,i,r] = 1
                
                # This is a system of 64 equations in U and W (bilinear)
                # We can linearize by fixing one variable at a time
                
                # Fix W, solve for U:
                # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
                # For fixed (k,i): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # Let C_r[i,j,k] = V[j,k,r]*W[k,i,r]
                # Then sum_r U[i,j,r]*C_r[i,j,k] = 1
                
                # This is linear in U! For each (i,j,k): sum_r U[i,j,r]*C_r[i,j,k] = 1
                # We have 64 equations and nm*target_rank = 8*target_rank unknowns
                
                # Build the linear system
                A_mat = np.zeros((n*m*p, nm * target_rank))
                b_vec = np.ones(n*m*p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat[eq_idx, i*m+j*target_rank + r] = V[j*p+k, r] * W[k*n+i, r]
                            eq_idx += 1
                
                # Solve least squares
                if A_mat.shape[1] >= A_mat.shape[0]:
                    U_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U = U_new.reshape(nm, target_rank)
                
                # Fix U, solve for W:
                # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # For fixed (i,j): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                # Let D_r[i,j,k] = U[i,j,r]*V[j,k,r]
                # Then sum_r D_r[i,j,k]*W[k,i,r] = 1
                
                A_mat2 = np.zeros((n*m*p, pn * target_rank))
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat2[eq_idx, k*n+i*target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
                            eq_idx += 1
                
                if A_mat2.shape[1] >= A_mat2.shape[0]:
                    W_new, _, _, _ = np.linalg.lstsq(A_mat2, b_vec, rcond=None)
                    W = W_new.reshape(pn, target_rank)
                
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residual += (val - 1)**2
                
                if residual < 1e-8:
                    # Found a valid decomposition!
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    # Convert to lists and return
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
    
    # Return the best we found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
