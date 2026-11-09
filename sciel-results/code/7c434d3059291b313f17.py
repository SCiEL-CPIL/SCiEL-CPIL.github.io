import numpy as np
from scipy.optimize import nnls
import time

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r has shape (n*m,), V_r has shape (m*p,), W_r has shape (p*n,)
    # In index form: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    
    # This is a bilinear system. Let's try to solve it for a given rank.
    
    # Approach: Fix U and V, solve for W (linear in W)
    # Or use iterative methods
    
    # Let's try rank from low to high
    best_result = None
    
    # First, try a few candidate ranks
    # Known: rank 90 works (schoolbook)
    # Try to find lower
    
    # Use the fact that for m×n × n×p, rank can be as low as O(m*p) when n is large
    # For 3×5 × 5×6, try rank = 30, 35, 40, etc.
    
    candidate_ranks = [24, 27, 30, 33, 36, 40, 45, 50, 60, 70, 80, 90]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of this rank
        # Use random initialization and iterative refinement
        np.random.seed(rank)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Iterative: fix two, solve for the third
        # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a 3-way tensor decomposition. Let's use a simple alternating least squares.
        
        converged = False
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            
            # Update W: for each (k,i), W[k*n+i, r] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This is: for each (k,i), and for each j, sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # Let A_{j,r} = U[i*m+j, r] * V[j*p+k, r]
            # Then sum_r A_{j,r} * W[k*n+i, r] = 1 for all j in [0,m)
            # This is a linear system: A * W[k*n+i, :] = ones(m)
            
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # Build the m x rank matrix A
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve A * w = ones(m)
                    try:
                        w, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = w
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Update U: for each (i,j), U[i*m+j, r] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # Let B_{k,r} = V[j*p+k, r] * W[k*n+i, r]
            # Then sum_r U[i*m+j, r] * B_{k,r} = 1 for all k in [0,p)
            
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        u, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = u
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Update V: for each (j,k), V[j*p+k, r] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # Let C_{i,r} = U[i*m+j, r] * W[k*n+i, r]
            # Then sum_r C_{i,r} * V[j*p+k, r] = 1 for all i in [0,n)
            
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    C = np.zeros((n, rank))
                    for i in range(n):
                        C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = v
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Check convergence
            if iteration % 10 == 9:
                # Compute the residual
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1))
                if max_err < 1e-6:
                    converged = True
                    break
        
        if converged:
            # Try to round to integers/half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded version still works
            max_err_rounded = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rounded[i * m + j, :] * V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                        max_err_rounded = max(max_err_rounded, abs(val - 1))
            
            if max_err_rounded < 1e-9:
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_rounded]
                V_list = [[float(x) for x in row] for row in V_rounded]
                W_list = [[float(x) for x in row] for row in W_rounded]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
            else:
                # Use the unrounded version if it's close enough
                if max_err < 1e-9:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
    
    # If no better rank found, fall back to schoolbook
    if best_result is None:
        rank = rank_schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
