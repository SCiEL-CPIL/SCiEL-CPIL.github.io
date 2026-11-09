"""Search for rank-minimal exact tensor decomposition of 3x5 x 5x5 matrix multiplication."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total = n * m * p  # 75
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Build tensor T as vector: T[idx] where idx = (i*m+j)*p+k maps to output (k*n+i)
    # Actually we need: for each (i,j,k), the output index is k*n+i, and the value is 1.
    # Let's build the linear system structure.
    # For a decomposition with rank r, we have r columns.
    # Equation for each (i,j,k): sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: fix r, randomly generate U and V columns, then solve for W.
    # The system is linear in W: for each output position o = k*n+i, 
    # and for each (i,j,k) that maps to o, we have an equation.
    # Actually each (i,j,k) gives one equation involving W at position k*n+i.
    # Let's reorganize: group equations by output position.
    # For output position o = k*n+i, the equations are:
    #   For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[o, c] = 1
    # So for fixed (i,k), we have m equations (one per j) that are all of the form:
    #   sum_c a_{j,c} * W[o,c] = 1  where a_{j,c} = U[i*m+j,c] * V[j*p+k,c]
    
    # This means: for each (i,k), the vector [W[o,0], ..., W[o,r-1]] must satisfy
    # a linear system A_{i,k} * w_o = 1 (vector of ones of length m)
    # where A_{i,k}[j, c] = U[i*m+j, c] * V[j*p+k, c]
    
    # So the approach:
    # 1. Choose r
    # 2. Randomly generate U (nm x r) and V (mp x r) with integer/half-integer entries
    # 3. For each (i,k), form A_{i,k} (m x r) and check if A_{i,k} * w = 1 has a solution
    #    If r >= m, we can solve. If r < m, the system is overdetermined.
    # 4. If all (i,k) systems are consistent, we have a valid decomposition.
    
    # Lower bound on rank: at least max(n, m, p) = 5? Actually the rank must be at least 
    # the dimension of the output space projected... The trivial lower bound is max(n*m, m*p, p*n) 
    # but that's not right. The rank must be at least the max of the "fiber" dimensions.
    # For each output position, we need to solve m equations in r unknowns, so r >= m = 5.
    # Also by symmetry considerations, rank >= max(n, p) = 5.
    # Let's try ranks from 5 upward.
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Generate schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def try_rank(r, max_iters=200):
        """Try to find a decomposition of rank r. Return (U, V, W) or None."""
        nonlocal start_time, deadline
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            # Generate random U and V with small integer entries
            # Use entries from {-2, -1, 0, 1, 2} to keep things simple
            U = np.random.randint(-2, 3, size=(nm, r))
            V = np.random.randint(-2, 3, size=(mp, r))
            
            # For each (i, k), check if the system is solvable
            W = np.zeros((pn, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build A_{i,k}: m x r matrix where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve A * w = 1 (vector of ones of length m)
                    # Use least squares and check residual
                    if r >= m:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if exact solution exists
                        if np.allclose(A @ w, np.ones(m), atol=1e-10):
                            W[k * n + i, :] = w
                        else:
                            valid = False
                            break
                    else:
                        # Overdetermined: check consistency
                        # The system A*w = 1 is consistent iff 1 is in the column space of A
                        # i.e., rank(A) == rank([A | 1])
                        A_aug = np.column_stack([A, np.ones(m)])
                        rank_A = np.linalg.matrix_rank(A)
                        rank_aug = np.linalg.matrix_rank(A_aug)
                        if rank_A != rank_aug:
                            valid = False
                            break
                        else:
                            w, residuals, rank_A2, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k * n + i, :] = w
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                # Check: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            error += abs(val - 1.0)
                if error < 1e-8:
                    # Convert to lists with integer/half-integer rounding
                    U_list = [[round(float(U[i, c]), 6) for c in range(r)] for i in range(nm)]
                    V_list = [[round(float(V[i, c]), 6) for c in range(r)] for i in range(mp)]
                    W_list = [[round(float(W[i, c]), 6) for c in range(r)] for i in range(pn)]
                    # Clean up near-integers
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for idx, val in enumerate(row):
                                if abs(val - round(val)) < 1e-6:
                                    row[idx] = int(round(val))
                                elif abs(val * 2 - round(val * 2)) < 1e-6:
                                    row[idx] = val * 2 / 2  # half-integer
                    return U_list, V_list, W_list
        return None
    
    # Try ranks from low to high
    for r in range(5, best_rank):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=50)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break  # Found minimal rank, stop
    
    return best_result
