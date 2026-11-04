"""Matrix multiplication 3x3x7 tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline
    
    # The naive rank is n*m*p = 63
    # Try to find lower rank decompositions
    
    # First, let's try known small ranks
    # For 3x3x3, rank is 23. For 3x3x7, we can try various ranks.
    
    # The tensor T[i,j,k] = delta_{i,j,k} for the multiplication
    # Actually, the multiplication tensor has entries T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k in the appropriate ranges
    
    # Let me set up the problem correctly
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # where the "k" index here corresponds to the column of B and row of C
    
    # Wait, let me re-read the problem. The output factor W[k*n+i,r] follows the trace convention.
    # So W has shape (p*n, rank) and the index is k*n+i where k is the column index of B/row of C, i is the row index of A/column of C.
    
    # The decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't look right. Let me think again.
    
    # Actually, for matrix multiplication C = A * B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Hmm, that's not quite right either.
    
    # Let me look at the seed code more carefully:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # So the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give the coefficient of A[i,j]*B[j,k] in C[i,k]
    # Which is 1 for the matching j.
    
    # So the condition is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this term contributes to C[i,k]
    # But actually, each (i,j,k) triple corresponds to one term A[i,j]*B[j,k] which contributes to C[i,k].
    # The tensor entry T[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition requires: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Wait, that's not right. The tensor is T[i,j,k] = 1, and we want:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 63 equations in r variables per matrix (total 3*r*r variables).
    
    # Let me try to find a rank-r decomposition by solving this system.
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        # We have U (n*m x r), V (m*p x r), W (p*n x r)
        # Constraint: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a bilinear/trilinear system. Let me try random initialization and iterative refinement.
        
        # Initialize randomly
        rng = np.random.RandomState(42)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Scale to make entries O(1)
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        # Use iterative method: fix two matrices, solve for the third
        # This is like alternating least squares for tensor decomposition
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Fix V and W, solve for U
            # For each (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This is a linear system in U[i*m+j, :]
            
            # Actually, let me think of it differently.
            # The constraint is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Fix V and W. For each i, j:
            # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k in [0,p)
            # Let's define M[i,j] as a p x r matrix where M[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we need: U[i*m+j, :] @ M[i,j] = ones(p)
            # This is an underdetermined system (r > p typically), so we can use least squares.
            
            U_new = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    # Build M[i,j]
                    M = np.zeros((p, r))
                    for k in range(p):
                        for r_idx in range(r):
                            M[k, r_idx] = V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    # Solve U[i*m+j, :] @ M = ones(p)
                    # This is: M.T @ U[i*m+j, :].T = ones(p)
                    # Use least squares
                    try:
                        U[i * m + j], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    except:
                        U[i * m + j] = U_new[i * m + j]
            
            # Similarly for V and W
            V_new = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        for r_idx in range(r):
                            M[i, r_idx] = U[i * m + j, r_idx] * W[k * n + i, r_idx]
                    try:
                        V[j * p + k], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    except:
                        V[j * p + k] = V_new[j * p + k]
            
            W_new = np.zeros((p * n, r))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        for r_idx in range(r):
                            M[j, r_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                    try:
                        W[k * n + i], residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    except:
                        W[k * n + i] = W_new[k * n + i]
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-10:
                break
            
            # Damping
            U = 0.5 * U + 0.5 * U_new
            V = 0.5 * V + 0.5 * V_new
            W = 0.5 * W + 0.5 * W_new
        
        return best_error, best_U, best_V, best_W
    
    # Try ranks from low to high
    # The trivial rank is 63. Let's try lower.
    # Known: 3x3x3 has rank 23. For 3x3x7, maybe we can do around 23*7/something?
    # Let's try ranks: 40, 45, 50, 55, 60, 63
    
    best_rank = 63
    best_result = None
    
    for r in [63, 60, 55, 50, 45, 40, 35, 30]:
        if time.time() > deadline - 10:
            break
        
        error, U, V, W = try_rank(r)
        
        if error < 1e-8:
            # Found a valid decomposition
            # Convert to lists
            U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U_list[i * m + j][r_idx] * V_list[j * p + k][r_idx] * W_list[k * n + i][r_idx]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = r
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    if best_result is None:
        # Fall back to naive
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
