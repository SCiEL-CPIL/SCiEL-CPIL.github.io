"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Construction 1: rank = n*m = 12
    # U[i*m+j, r] = 1 if r == i*m+j else 0
    # V[j*p+k, r] = 1 for all
    # W[k*n+i, r] = 1 for all
    rank12 = n * m
    U12 = np.zeros((n * m, rank12))
    V12 = np.ones((m * p, rank12))
    W12 = np.ones((p * n, rank12))
    for i in range(n):
        for j in range(m):
            U12[i * m + j, i * m + j] = 1
    
    # Verify rank 12
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if verify(U12, V12, W12, rank12):
        # Try to find lower rank
        best_U, best_V, best_W, best_rank = U12, V12, W12, rank12
        
        # Try ranks from 5 upward
        for target_rank in range(5, rank12):
            if time.time() > deadline - 20:
                break
            
            found = False
            for seed in range(100):
                if time.time() > deadline - 20:
                    break
                
                np.random.seed(seed * 7919 + target_rank)
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # Alternating least squares
                for it in range(500):
                    if time.time() > deadline - 20:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            A = np.zeros((target_rank, p))
                            for r in range(target_rank):
                                for k in range(p):
                                    A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                            try:
                                U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                            except:
                                pass
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            A = np.zeros((target_rank, n))
                            for r in range(target_rank):
                                for i in range(n):
                                    A[r, i] = U[i*m+j, r] * W[k*n+i, r]
                            try:
                                V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                            except:
                                pass
                    
                    # Update W
                    for i in range(n):
                        for k in range(p):
                            A = np.zeros((target_rank, m))
                            for r in range(target_rank):
                                for j in range(m):
                                    A[r, j] = U[i*m+j, r] * V[j*p+k, r]
                            try:
                                W[k*n+i, :] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                            except:
                                pass
                    
                    # Check
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                                max_err = max(max_err, abs(val - 1))
                    
                    if max_err < 1e-12:
                        found = True
                        break
                
                if found:
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    break
            
            if found:
                break
        
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
