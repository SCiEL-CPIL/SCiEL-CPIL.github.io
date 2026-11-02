"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    def compute_error(U, V, W):
        """Compute max absolute error of the tensor decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_rank_with_restart(rank, max_iters=400, num_restarts=3):
        """Try to find decomposition with multiple random restarts."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + restart * 7919)
            
            # Initialize with schoolbook + perturbation
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            num_schoolbook = min(rank, n * m * p)
            for c in range(num_schoolbook):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                if i < n and j < m and k < p:
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
            
            # Add random perturbation
            U += np.random.randn(n * m, rank) * 0.3
            V += np.random.randn(m * p, rank) * 0.3
            W += np.random.randn(p * n, rank) * 0.3
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = V[j*p:(j+1)*p, :] * W[np.arange(p) * n + i, :][np.newaxis, :]
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = U[np.arange(n) * m + j, :] * W[k*n:(k+1)*n, :].T
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = U[i*m:(i+1)*m, :].T * V[np.arange(m) * p + k, :].T
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = x
                        except:
                            pass
                
                # Check error periodically
                if iteration % 20 == 0 or iteration == max_iters - 1:
                    err = compute_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-12:
                        return U, V, W, err
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from low to high
    best_result = None
    best_rank = n * m * p
    
    for rank in range(36, min(145, n*m*p + 1)):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_restart(rank, max_iters=300, num_restarts=2)
        if result is not None:
            U, V, W, err = result
            if err < 1e-10:
                best_result = (U, V, W)
                best_rank = rank
                break
            elif best_result is None:
                best_result = (U, V, W)
                best_rank = rank
    
    if best_result is not None and best_rank < n * m * p:
        U, V, W = best_result
        U_list = [[round(float(U[i,j]), 10) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[round(float(V[i,j]), 10) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[round(float(W[i,j]), 10) for j in range(W.shape[1])] for i in range(W.shape[0])]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
