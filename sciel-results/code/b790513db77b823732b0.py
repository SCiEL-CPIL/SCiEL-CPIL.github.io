import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 2, 7
    rank_initial = 25
    
    # Build initial rank-25 solution
    U = np.zeros((n*m, rank_initial))
    V = np.zeros((m*p, rank_initial))
    W = np.zeros((p*n, rank_initial))
    
    def add_strassen(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    # Try to reduce rank
    best_rank = rank_initial
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Build target tensor
    # T[i,k,j] = 1 for all valid i,k,j
    # U has shape (4, rank), V has shape (14, rank), W has shape (14, rank)
    # C[i,j] = sum_r U[i,r] * (V[:,r] @ B) ... 
    
    # Actually, let me set up the equations properly.
    # The tensor equation is: for all (i, k, j),
    # sum_r U[i*2+k, r] * V[k*7+j, r] * W[j*2+i, r] = 1
    
    # Wait, I need to be careful about index conventions.
    # U has shape (n*m, rank) = (4, rank). Index: i*m + k = i*2 + k
    # V has shape (m*p, rank) = (14, rank). Index: k*p + j = k*7 + j  
    # W has shape (p*n, rank) = (14, rank). Index: j*n + i = j*2 + i
    
    # The equation: sum_r U[i*2+k, r] * V[k*7+j, r] * W[j*2+i, r] = 1 for all i,k,j
    
    def check_solution(U, V, W):
        """Check if the decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err
    
    def try_reduce_rank(U0, V0, W0, target_rank, max_iters=200, tol=1e-10):
        """Try to find a decomposition of target_rank by zeroing out columns."""
        if target_rank >= U0.shape[1]:
            return U0.copy(), V0.copy(), W0.copy()
        
        # Zero out the last (U0.shape[1] - target_rank) columns
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        for r in range(target_rank, U0.shape[1]):
            U[:, r] = 0
            V[:, r] = 0
            W[:, r] = 0
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix V and W, solve for U
            # For each (i,k): sum_r U[i*2+k, r] * V[k*7+j, r] * W[j*2+i, r] = 1 for all j
            # This is: sum_r U[i*2+k, r] * (V[k*7+j, r] * W[j*2+i, r]) = 1 for all j
            # Let M_U[i*2+k, r, j] = V[k*7+j, r] * W[j*2+i, r]
            # Then: U[i*2+k, :] @ M_U[i*2+k, :, j] = 1 for all j
            # This is a linear system: U[i*2+k, :] @ A_i_k = 1
            # where A_i_k has shape (rank, p) with A_i_k[r, j] = V[k*p+j, r] * W[j*n+i, r]
            
            for i in range(n):
                for k in range(m):
                    # Build A matrix: shape (target_rank, p)
                    A = np.zeros((target_rank, p))
                    for r in range(target_rank):
                        for j in range(p):
                            A[r, j] = V[k*p+j, r] * W[j*n+i, r]
                    # Solve U[i*2+k, :target_rank] @ A = 1
                    # A has shape (target_rank, p), we want U_row @ A = ones(p)
                    # So: A^T @ U_row^T = ones(p)
                    if target_rank >= p:
                        # Overdetermined, use least squares
                        U[i*2+k, :target_rank] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    else:
                        # Underdetermined, use least squares (minimum norm)
                        U[i*2+k, :target_rank] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    U[i*2+k, target_rank:] = 0
            
            # Fix U and W, solve for V
            for k in range(m):
                for j in range(p):
                    A = np.zeros((target_rank, n))
                    for r in range(target_rank):
                        for i in range(n):
                            A[r, i] = U[i*m+k, r] * W[j*n+i, r]
                    V[k*p+j, :target_rank] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    V[k*p+j, target_rank:] = 0
            
            # Fix U and V, solve for W
            for j in range(p):
                for i in range(n):
                    A = np.zeros((target_rank, m))
                    for r in range(target_rank):
                        for k in range(m):
                            A[r, k] = U[i*m+k, r] * V[k*p+j, r]
                    W[j*n+i, :target_rank] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                    W[j*n+i, target_rank:] = 0
        
        return U, V, W
    
    # Try reducing from 25 down
    for target_rank in range(24, 19, -1):
        if time.time() > deadline - 30:
            break
        
        U_try, V_try, W_try = try_reduce_rank(U, V, W, target_rank, max_iters=100)
        err = check_solution(U_try, V_try, W_try)
        
        if err < 1e-8:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            print(f"Found rank {target_rank} solution with error {err}")
            break
    
    if best_rank < rank_initial:
        # Trim to best_rank
        U_out = best_U[:, :best_rank]
        V_out = best_V[:, :best_rank]
        W_out = best_W[:, :best_rank]
    else:
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
