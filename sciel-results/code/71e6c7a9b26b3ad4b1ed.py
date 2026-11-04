"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T[i, j, k] = 1 if (i,j,k) is valid
    # T is shape (n, m, p) where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The tensor T is all ones of shape (n, m, p)
    T = np.ones((n, m, p))
    
    # Reshape to a matrix for SVD: flatten (i,j) vs k, but that's not quite right
    # The tensor decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This is a 3-way tensor decomposition with specific index couplings
    
    # Let's use a greedy approach: start with schoolbook and try to reduce rank
    # by finding linear dependencies
    
    # First, let's get the schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition
    # Use randomized tensor decomposition
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        # Reconstruct T
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            # U[:, c] is shape (n*m,), V[:, c] is shape (m*p,), W[:, c] is shape (p*n,)
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
        return np.allclose(T_recon, T, atol=1e-9)
    
    # Try to find a low-rank decomposition using random initialization and optimization
    best_rank = rank_school
    best_result = None
    
    # Start from rank 20 and try to go lower
    for target_rank in range(20, 81):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Use ALS (Alternating Least Squares) to fit the tensor
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T[i,j,k] for all k
                # This is: U[i*m+j, :] * (V[j*p+0:m*p, :] * W[0:p*n, :] reshaped) = T[i,j,:]
                
                # For fixed (i,j), we need: sum_c U[i*m+j,c] * (sum_k V[j*p+k,c] W[k*n+i,c] * T[i,j,k])
                # Actually, let's set up the linear system properly
                
                # For U update: fix V and W, solve for U
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ M where M[c,k] = V[j*p+k,c] * W[k*n+i,c]
                # So U[i*m+j,:] = T[i,j,:] @ M^{-1} (or least squares)
                
                for i in range(n):
                    for j in range(m):
                        # M shape: (target_rank, p) where M[c,k] = V[j*p+k,c] * W[k*n+i,c]
                        M = np.zeros((target_rank, p))
                        for k in range(p):
                            for c in range(target_rank):
                                M[c, k] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve: U[i*m+j, :] @ M = T[i, j, :]
                        # i.e., M.T @ U[i*m+j, :].T = T[i, j, :].T
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                        except:
                            pass
                
                # Update V: fix U and W
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed (j,k): T[:,j,k] = V[j*p+k,:] @ N where N[c,i] = U[i*m+j,c] * W[k*n+i,c]
                        N = np.zeros((target_rank, n))
                        for i in range(n):
                            for c in range(target_rank):
                                N[c, i] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(N.T, T[:, j, k], rcond=None)[0]
                        except:
                            pass
                
                # Update W: fix U and V
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # For fixed (k,i): T[i,:,k] = W[k*n+i,:] @ P where P[c,j] = U[i*m+j,c] * V[j*p+k,c]
                        P = np.zeros((target_rank, m))
                        for j in range(m):
                            for c in range(target_rank):
                                P[c, j] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(P.T, T[i, :, k], rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if verify(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = (U.copy(), V.copy(), W.copy())
                    break
        
        if best_result is not None and best_rank <= target_rank:
            break
    
    # If no better result found, return schoolbook
    if best_result is None:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

