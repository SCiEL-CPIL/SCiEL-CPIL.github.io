import numpy as np
import time
from fractions import Fraction
from itertools import product
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Helper to verify a decomposition
    def verify(U_mat, V_mat, W_mat, r):
        tensor = np.zeros((n, m, p))
        for c in range(r):
            uc = U_mat[:, c].reshape(n, m)
            vc = V_mat[:, c].reshape(m, p)
            wc = W_mat[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        return np.allclose(tensor, 1.0, atol=1e-9)

    # First, try to find a rank-17 decomposition for the 2x2x5 sub-problem
    # The full problem is 2x3x5.
    # Split m=3 into m1=2 and m2=1.
    # T = T1 (2x2x5) + T2 (2x1x5).
    # Rank(T2) = 10 (minimal).
    # If we find Rank(T1) = 17, Total Rank = 27.
    
    # Construct T1 tensor (2x2x5)
    n1, m1, p1 = 2, 2, 5
    T1 = np.zeros((n1, m1, p1))
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                T1[i, j, k] = 1.0

    # Search for rank-17 decomposition of T1
    # We will use a randomized restart with gradient descent
    best_rank17_U = None
    best_rank17_V = None
    best_rank17_W = None
    
    # We know Strassen gives rank 7 for 2x2x2.
    # We can initialize with Strassen-like structures.
    
    def find_rank17_decomp():
        # Try to find 17 rank-1 terms
        # We can use the 7 terms from Strassen for k=0,1 and 7 for k=2,3
        # and try to fit the k=4 part with 3 terms that also have small components in k=0..3?
        # Or just do a full search.
        
        # Initialize with 17 random terms
        # Or use the 18-term decomposition and try to merge two?
        
        # Let's try a specific known construction if possible, or a search.
        # Given time constraints, a robust search is better.
        
        # We will try to solve for 17 terms.
        # Variables: U (2*2*17), V (2*5*17), W (5*2*17). Total 136 vars.
        # Equations: 2*2*5 = 20 equations.
        # This is underdetermined, so a solution should exist if rank <= 17.
        
        # We use a simple iterative solver:
        # 1. Initialize U, V, W with random values.
        # 2. Iteratively update one matrix while holding others fixed.
        #    e.g., fix V, W. Solve for U:
        #    T1 = sum_c U_c V_c W_c
        #    For each (i,j), T1[i,j,:] = sum_c U_c[i,j] (V_c W_c)_elem
        #    This is a linear system for U_c[i,j].
        
        max_iters = 50
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            U = np.random.randn(n1*m1, 17) * 0.1
            V = np.random.randn(m1*p1, 17) * 0.1
            W = np.random.randn(p1*n1, 17) * 0.1
            
            # Add some structure: copy Strassen terms for first 14
            # This might help convergence
            # But random restarts might be better to avoid local minima
            
            for it in range(max_iters):
                # Update U
                # T1[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # For fixed V, W, this is linear in U.
                # Reshape T1 to (n1*m1, p1) ? No, T1 is (n1, m1, p1).
                # We can flatten i,j.
                # T1_flat (n1*m1, p1) = sum_c U_flat[:, c] * (V_c * W_c)_flattened?
                # No, V_c is (m1, p1), W_c is (p1, n1).
                # The term is U_c (n1, m1) * V_c (m1, p1) * W_c (p1, n1) ?
                # No, the tensor product is U_c[i,j] * V_c[j,k] * W_c[k,i].
                # For a fixed c, the contribution to T1[i,j,k] is U_c[i,j] V_c[j,k] W_c[k,i].
                # We want to find U_c such that sum_c ... = T1.
                
                # Let's update U.
                # For each c, U_c is (n1, m1).
                # The equation is:
                # sum_c U_c[i,j] V_c[j,k] W_c[k,i] = T1[i,j,k]
                # This is a linear system for U_c[i,j].
                # Let A_c be a matrix of size (n1*m1, p1*n1) ?
                # Vectorize U_c to u_c (4, 1).
                # Vectorize T1 to t (20, 1).
                # The operator L_c(u_c) is a vector of size 20.
                # (L_c(u_c))[k, i, j] = u_c[i,j] * V_c[j,k] * W_c[k,i].
                # We can build a matrix M of size (20, 4*17) where columns are the linear operators for each U_c[i,j].
                # This is too large to build every iteration? 20 x 68. It's small.
                
                # Build M_U
                M_U = np.zeros((n1*m1*p1, n1*m1*17))
                for c in range(17):
                    for i in range(n1):
                        for j in range(m1):
                            col_idx = c * (n1*m1) + (i*m1 + j)
                            for k in range(p1):
                                row_idx = (i*m1*p1) + (j*p1) + k
                                M_U[row_idx, col_idx] = V_c[j, k] * W_c[k, i]
                
                # Solve M_U * u_vec = t_vec
                t_vec = T1.flatten()
                try:
                    u_vec, residuals, rank, sv = np.linalg.lstsq(M_U, t_vec, rcond=None)
                except:
                    continue
                    
                U = u_vec.reshape(17, n1*m1).T # Shape (n1*m1, 17)
                
                # Similarly update V and W
                # Update V
                M_V = np.zeros((n1*m1*p1, m1*p1*17))
                for c in range(17):
                    for j in range(m1):
                        for k in range(p1):
                            col_idx = c * (m1*p1) + (j*p1 + k)
                            for i in range(n1):
                                row_idx = (i*m1*p1) + (j*p1) + k
                                M_V[row_idx, col_idx] = U[i, j, c] * W[k, i, c]
                
                try:
                    v_vec, residuals, rank, sv = np.linalg.lstsq(M_V, t_vec, rcond=None)
                except:
                    continue
                V = v_vec.reshape(17, m1*p1).T
                
                # Update W
                M_W = np.zeros((n1*m1*p1, p1*n1*17))
                for c in range(17):
                    for k in range(p1):
                        for i in range(n1):
                            col_idx = c * (p1*n1) + (k*n1 + i)
                            for j in range(m1):
                                row_idx = (i*m1*p1) + (j*p1) + k
                                M_W[row_idx, col_idx] = U[i, j, c] * V[j, k, c]
                
                try:
                    w_vec, residuals, rank, sv = np.linalg.lstsq(M_W, t_vec, rcond=None)
                except:
                    continue
                W = w_vec.reshape(17, p1*n1).T
                
                # Check error
                err = 0
                for i in range(n1):
                    for j in range(m1):
                        for k in range(p1):
                            val = 0
                            for c in range(17):
                                val += U[i, j, c] * V[j, k, c] * W[k, i, c]
                            err += (val - T1[i, j, k])**2
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    # Check if we can round to half-integers
                    # Try rounding to nearest 0.5
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    err_round = 0
                    for i in range(n1):
                        for j in range(m1):
                            for k in range(p1):
                                val = 0
                                for c in range(17):
                                    val += U_round[i, j, c] * V_round[j, k, c] * W_round[k, i, c]
                                err_round += (val - T1[i, j, k])**2
                    if err_round < 1e-6:
                        return U_round, V_round, W_round
                        
        if best_err < 1e-6:
            # Try rounding the best found
            U_round = np.round(best_U * 2) / 2
            V_round = np.round(best_V * 2) / 2
            W_round = np.round(best_W * 2) / 2
            err_round = 0
            for i in range(n1):
                for j in range(m1):
                    for k in range(p1):
                        val = 0
                        for c in range(17):
                            val += U_round[i, j, c] * V_round[j, k, c] * W_round[k, i, c]
                        err_round += (val - T1[i, j, k])**2
            if err_round < 1e-6:
                return U_round, V_round, W_round
                
        return None

    # Run the search
    decomp17 = find_rank17_decomp()
    
    if decomp17 is not None:
        U17, V17, W17 = decomp17
        # Construct full rank-27 decomposition
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Fill T1 part (j=0,1)
        # U17 is (2*2, 17) -> U[i, j, c] for i in 0..1, j in 0..1
        for i in range(2):
            for j in range(2):
                for c in range(17):
                    U[i*m+j, c] = U17[i, j, c]
        
        for j in range(2):
            for k in range(5):
                for c in range(17):
                    V[j*p+k, c] = V17[j, k, c]
                    
        for k in range(5):
            for i in range(2):
                for c in range(17):
                    W[k*n+i, c] = W17[k, i, c]
                    
        # Fill T2 part (j=2)
        # Rank 10 schoolbook
        col = 17
        for i in range(2):
            for k in range(5):
                U[i*m+2, col] = 1
                V[2*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
        # Verify
        if verify(U, V, W, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 28 if rank 27 search fails
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    strassen_block(0, 1, 0, 1, 0, 1)
    strassen_block(0, 1, 0, 1, 2, 3)
    schoolbook_block([0, 1], [0, 1], [4])
    schoolbook_block([0, 1], [2], [0, 1])
    schoolbook_block([0, 1], [2], [2, 3])
    schoolbook_block([0, 1], [2], [4])

    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: Rank 30
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
