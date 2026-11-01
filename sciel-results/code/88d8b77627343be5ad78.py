"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 125
    
    # Try ranks from 15 upward
    for rank in range(15, 46):
        if time.time() > deadline:
            break
        result = _find_decomposition(n, m, p, rank)
        if result is not None:
            return result
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, rank):
    for seed in range(50):
        if time.time() > 120:
            return None
        np.random.seed(seed * 7919 + 1)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Coordinate descent
        for it in range(300):
            # Update U rows
            for idx in range(n*m):
                i, j = divmod(idx, m)
                # Gradient: for each r, d/dr sum_k W[k*n+i,r]*V[j*p+k,r] * U[idx,r]
                # We want sum_r U[idx,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                # Fix V,W, solve for U[idx,:]
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                # Solve A @ U[idx,:] = b in least squares
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[idx, :] = sol
            
            # Update V rows
            for idx in range(m*p):
                j, k = divmod(idx, p)
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[idx, :] = sol
            
            # Update W rows
            for idx in range(p*n):
                k, i = divmod(idx, n)
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[idx, :] = sol
        
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _check(n, m, p, U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _check(n, m, p, U, V, W):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0) > 1e-6:
                    return False
    return True
# EVOLVE-BLOCK-END
