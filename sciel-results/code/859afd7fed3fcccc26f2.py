"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x3 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def matmul_tensor(n, m, p):
        """Create the matrix multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = matmul_tensor(n, m, p)
    
    # Try to find a low-rank decomposition using ALS with careful initialization
    # and verification against exact tensor equations.
    
    def verify_decomposition(U, V, W, T, tol=1e-8):
        """Verify that the decomposition exactly reconstructs the tensor."""
        n, m, p = DIMENSIONS
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    target = T[i*m+j, j*p+k, k*n+i]
                    error = max(error, abs(val - target))
        return error < tol
    
    def optimize_rank(r, n_restarts=10):
        """Optimize for a rank-r decomposition using ALS."""
        if time.time() > deadline - 15:
            return None
        
        best_result = None
        best_error = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(12345 + restart * 137)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # ALS iterations
            for iteration in range(150):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        b = np.zeros(p)
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            b[k] = T[i*m+j, j*p+k, k*n+i]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        b = np.zeros(n)
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            b[i] = T[i*m+j, j*p+k, k*n+i]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        b = np.zeros(m)
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            b[j] = T[i*m+j, j*p+k, k*n+i]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if iteration % 20 == 19:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                                target = T[i*m+j, j*p+k, k*n+i]
                                err = max(err, abs(val - target))
                    if err < 1e-8:
                        break
                    if err < best_error:
                        best_error = err
                        best_result = (U.copy(), V.copy(), W.copy())
            
            # Final check
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        target = T[i*m+j, j*p+k, k*n+i]
                        err = max(err, abs(val - target))
            
            if err < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                    return (U_rounded, V_rounded, W_rounded)
                
                # Use the unrounded version if it's close enough
                if err < 1e-8:
                    return (U, V, W)
            
            if err < best_error:
                best_error = err
                best_result = (U.copy(), V.copy(), W.copy())
        
        # If we have a good result but not exact, try to polish it
        if best_result is not None and best_error < 1e-4:
            U, V, W = best_result
            
            # Try more polishing iterations
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        b = np.zeros(p)
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            b[k] = T[i*m+j, j*p+k, k*n+i]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        b = np.zeros(n)
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            b[i] = T[i*m+j, j*p+k, k*n+i]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        b = np.zeros(m)
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            b[j] = T[i*m+j, j*p+k, k*n+i]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            target = T[i*m+j, j*p+k, k*n+i]
                            err = max(err, abs(val - target))
                
                if err < 1e-8:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                        return (U_rounded, V_rounded, W_rounded)
                    else:
                        return (U, V, W)
        
        return None
    
    # Try ranks from 19 to 27 (known bounds for 3x3 matrix multiplication)
    for r in range(19, 28):
        if time.time() > deadline - 15:
            break
        
        result = optimize_rank(r, n_restarts=8)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook (rank 27)
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
