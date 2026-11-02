import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def check_valid(U, V, W, rank, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Strassen decomposition for 3x3x3 matrix multiplication
    # U: (9, 7), V: (9, 7), W: (9, 7)
    # Strassen's 7 multiplications for 2x2x2, extended to 3x3x3
    
    # The 7 Strassen products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 3x3x3, we can use a similar approach. Let's use the known rank-23 decomposition
    # for 3x3x3 matrix multiplication. Actually, the minimal rank for 3x3x3 is not known exactly,
    # but we can use a decomposition with rank 23.
    
    # Let's use a simpler approach: embed the 2x2x2 Strassen into 3x3x3
    # and handle the remaining terms.
    
    # Actually, let's use a known rank-23 decomposition for 3x3x3.
    # The decomposition has 23 terms.
    
    # For simplicity, let's start with the schoolbook rank 27 for 3x3x3
    # and then extend to 3x3x5.
    
    # Schoolbook for 3x3x3: rank 27
    U_333 = np.zeros((9, 27))
    V_333 = np.zeros((9, 27))
    W_333 = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_333[i * 3 + j, col] = 1
                V_333[j * 3 + k, col] = 1
                W_333[k * 3 + i, col] = 1
                col += 1
    
    # Now extend to 3x3x5
    # The output is 3x5, so W has shape (5*3, rank) = (15, rank)
    # U has shape (3*3, rank) = (9, rank)
    # V has shape (3*5, rank) = (15, rank)
    
    # We need to handle the extra 2 columns of the output (k=3, k=4)
    # For the first 3 columns (k=0,1,2), we use the 3x3x3 decomposition
    # For columns k=3,4, we need additional terms
    
    # Let's use rank 27 for the first 3 columns and add terms for k=3,4
    # Actually, we can share some terms.
    
    # Let's try a direct approach: start with rank 45 (schoolbook for 3x3x5)
    # and try to reduce it.
    
    # Schoolbook for 3x3x5: rank 45
    rank_init = 45
    U = np.zeros((9, rank_init))
    V = np.zeros((15, rank_init))
    W = np.zeros((15, rank_init))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U[i * 3 + j, col] = 1
                V[j * 5 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    
    current_rank = rank_init
    
    def compute_residual_vec(params, rank):
        U_r = params[:9*rank].reshape((9, rank))
        V_r = params[9*rank:9*rank + 15*rank].reshape((15, rank))
        W_r = params[9*rank + 15*rank:].reshape((15, rank))
        
        residuals = np.zeros(3*3*5)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    val = np.dot(U_r[i*3+j, :], V_r[j*5+k, :] * W_r[k*3+i, :])
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals
    
    # Try greedy removal
    for it in range(30):
        if time.time() > deadline:
            break
        
        if not check_valid(U, V, W, current_rank, tol=1e-4):
            break
            
        removed = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            mask = [x for x in range(current_rank) if x != r]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            new_rank = current_rank - 1
            
            if check_valid(U_new, V_new, W_new, new_rank, tol=1e-4):
                U, V, W = U_new, V_new, W_new
                current_rank = new_rank
                removed = True
                break
            
            x0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            
            try:
                result = least_squares(
                    compute_residual_vec, x0, args=(new_rank,),
                    method='lm',
                    max_nfev=200,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                if result.cost < 1e-10:
                    U_opt = result.x[:9*new_rank].reshape((9, new_rank))
                    V_opt = result.x[9*new_rank:9*new_rank + 15*new_rank].reshape((15, new_rank))
                    W_opt = result.x[9*new_rank + 15*new_rank:].reshape((15, new_rank))
                    if check_valid(U_opt, V_opt, W_opt, new_rank, tol=1e-4):
                        U, V, W = U_opt, V_opt, W_opt
                        current_rank = new_rank
                        removed = True
                        break
            except Exception:
                pass
        
        if not removed:
            break
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
