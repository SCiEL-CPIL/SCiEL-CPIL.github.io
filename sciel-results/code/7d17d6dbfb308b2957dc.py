"""Find rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # Target: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from low to high
    # Schoolbook is n*m*p = 40
    # Try to find lower rank
    
    def try_rank(r, seed=42):
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Newton-like iteration: at each step, fix two matrices and solve for the third
        for it in range(200):
            # Fix V and W, solve for U
            # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1 for all k
            # This gives p equations per (i,j), with r unknowns
            # We want: for each (i,j,k): sum_c U[i*m+j,c] * A[k,c] = 1 where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
            # So: U[i*m+j,:] @ A.T = ones(p)
            # If r >= p, we can solve this linear system
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A.T @ U[i*m+j,:] = ones(p)
                    # i.e., U[i*m+j,:] = (A.T)^{-1} @ ones if square, else least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = max(err, abs(val - 1))
            if err < 1e-10:
                # Round to nearest half-integers or integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                # Verify
                err2 = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            err2 = max(err2, abs(val - 1))
                if err2 < 1e-9:
                    return U_rounded, V_rounded, W_rounded
                # Try integer rounding
                U_rounded = np.round(U)
                V_rounded = np.round(V)
                W_rounded = np.round(W)
                err2 = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            err2 = max(err2, abs(val - 1))
                if err2 < 1e-9:
                    return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from low to high
    for r in range(16, 41):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
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
