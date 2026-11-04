import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    # Target tensor T_{i,j,k}
    # We need sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k
    # This is equivalent to T_{i,j,k} = 1
    
    # Try to find lower rank
    # Lower bound is max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    # Try ranks from 27 to 53
    
    start_time = time.time()
    timeout = 120
    
    for R in range(27, R_school):
        if time.time() - start_time > timeout - 10:
            break
            
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # We want to solve for U, V, W such that
        # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1
        # This is a system of equations.
        # We can use a simple iterative method or least squares.
        # Let's try to fix W and solve for U, V?
        # Or use a gradient descent approach on the error.
        
        # Error function: E = sum_{i,j,k} (sum_r U_{ij,r} V_{jk,r} W_{ki,r} - 1)^2
        # This is expensive to compute directly for large R.
        # But n*m*p is small (54).
        
        # We can linearize by fixing two matrices.
        # Fix V and W, solve for U.
        # For each (i,j), we have sum_r U_{ij,r} (V_{jk,r} W_{ki,r}) = 1 for all k.
        # This is a linear system for U_{ij, :}.
        
        for iter in range(100):
            if time.time() - start_time > timeout - 5:
                break
                
            # Fix V, W. Solve for U.
            for i in range(n):
                for j in range(m):
                    # Eq: sum_r U_{ij,r} * (V_{j*p+k, r} * W_{k*n+i, r}) = 1 for all k
                    # Let M_{k, r} = V_{j*p+k, r} * W_{k*n+i, r}
                    # We want M @ U_{ij, :}^T = 1 (vector of ones of length p)
                    M = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            M[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    # Solve M @ x = 1
                    # Least squares
                    x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i * m + j, :] = x
                    
            # Fix U, W. Solve for V.
            for j in range(m):
                for k in range(p):
                    # Eq: sum_r V_{jk,r} * (U_{i*m+j, r} * W_{k*n+i, r}) = 1 for all i
                    # Let M_{i, r} = U_{i*m+j, r} * W_{k*n+i, r}
                    # We want M @ V_{jk, :}^T = 1 (vector of ones of length n)
                    M = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j * p + k, :] = x
                    
            # Fix U, V. Solve for W.
            for k in range(p):
                for i in range(n):
                    # Eq: sum_r W_{ki,r} * (U_{i*m+j, r} * V_{j*p+k, r}) = 1 for all j
                    # Let M_{j, r} = U_{i*m+j, r} * V_{j*p+k, r}
                    # We want M @ W_{ki, :}^T = 1 (vector of ones of length m)
                    M = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k * n + i, :] = x
            
            # Check error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err += (val - 1)**2
            if err < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
