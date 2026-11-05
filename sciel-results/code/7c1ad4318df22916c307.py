"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline guard
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # The matrix multiplication tensor T[i,j,k] = 1 if i,j,k are valid indices
    # We need to decompose T as sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r], so the third factor is indexed by (k, i)
    # Let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor is T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # And this must equal the matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (since we're decomposing the multiplication tensor itself)
    
    # Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (A*B)[i,k] = sum_j A[i,j] * B[j,k], and each term A[i,j]*B[j,k] corresponds to the tensor entry (i,j,k)
    
    # Let's use ALS to find a rank-r decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For a given rank r, we have r * (n*m + m*p + p*n) parameters
    # We need to satisfy n*m*p = 45 equations
    
    # ALS approach:
    # Fix V and W, solve for U (linear least squares)
    # Fix U and W, solve for V (linear least squares)
    # Fix U and V, solve for W (linear least squares)
    
    # Let's try ranks from 44 down to 30
    max_rank = 44
    min_rank = 30
    num_restarts = 5
    
    for r in range(max_rank, min_rank - 1, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            random.seed(restart * 1000 + r)
            np.random.seed(restart * 1000 + r)
            
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) = 1
                # For fixed (i,j), we have p equations (one for each k) in r unknowns (U[i*m+j, 0..r-1])
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # U[i*m+j, :] @ A = ones(p)
                
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        # Solve least squares: A.T @ U[i*m+j, :] = b
                        # Or: (A.T @ A) @ U[i*m+j, :] = A.T @ b
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for V: for each (j,k), V[j*p+k, :] should satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # For fixed (j,k), we have n equations in r unknowns
                # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # V[j*p+k, :] @ B = ones(n)
                
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, r))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B.T, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for W: for each (k,i), W[k*n+i, :] should satisfy
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # For fixed (k,i), we have m equations in r unknowns
                # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # W[k*n+i, :] @ C = ones(m)
                
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, r))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C.T, b, rcond=None)[0]
                        except:
                            pass
            
            # Check the quality of the decomposition
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Found a valid decomposition!
                # Convert to list format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Check if values are close to integers or half-integers
                # If so, snap them
                U_snapped = [[round(2 * x) / 2 for x in row] for row in U_list]
                V_snapped = [[round(2 * x) / 2 for x in row] for row in V_list]
                W_snapped = [[round(2 * x) / 2 for x in row] for row in W_list]
                
                # Verify the snapped values still work
                max_err_snapped = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_idx in range(r):
                                val += U_snapped[i * m + j][r_idx] * V_snapped[j * p + k][r_idx] * W_snapped[k * n + i][r_idx]
                            err = abs(val - 1.0)
                            if err > max_err_snapped:
                                max_err_snapped = err
                
                if max_err_snapped < 1e-6:
                    best_result = {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                    best_rank = r
                    found = True
                    break
                else:
                    # Use the non-snapped values if they're close enough
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # We found a decomposition at rank r, try lower ranks
            continue
        else:
            # If we couldn't find a valid decomposition at this rank, stop
            # But continue trying lower ranks in case initialization matters
            pass
    
    return best_result
# EVOLVE-BLOCK-END
