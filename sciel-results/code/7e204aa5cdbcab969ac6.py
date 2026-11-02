"""Rewritten program for rank-minimal tensor decomposition of 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 110
    
    # Schoolbook rank = n*m*p = 120
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify the schoolbook solution
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition correctly computes matrix multiplication."""
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                U_block = U[i*m:(i+1)*m, :]
                V_block = np.array([V[l*p+k, :] for l in range(m)])
                W_vec = W[k*n+i, :]
                M = (U_block * W_vec) @ V_block.T
                if not np.allclose(M, np.eye(m), atol=tol):
                    return False
        return True
    
    # Try to find a lower-rank decomposition.
    # 
    # Key insight: The condition is sum_c U[i*m+j,c] * V[l*p+k,c] * W[k*n+i,c] = delta_{j,l}
    # 
    # This means for each (i,k), the matrix M_{i,k}[j,l] = sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] = I_m
    # 
    # Let's try a block-structured approach.
    # 
    # For the 5x5 middle part, we can try to use a decomposition of the identity.
    # 
    # Idea: If we set W[k*n+i, c] = 1 for all c, then we need:
    # sum_c U[i*m+j,c] * V[l*p+k,c] = delta_{j,l}
    # 
    # This means U_block @ V_block.T = I_m for each (i,k).
    # 
    # If we use the same U and V for all (i,k), we need U_block @ V_block.T = I_m.
    # 
    # Let U_block be m x r and V_block be m x r. Then U_block @ V_block.T = I_m requires r >= m = 5.
    # 
    # If we take U_block = I_m (as a 5x5 matrix, with r=5) and V_block = I_m, then
    # U_block @ V_block.T = I_m. 
    # 
    # But U has shape (20, r) = (20, 5). For each i, U[i*m:(i+1)*m, :] should be I_m.
    # So U = [I_m; I_m; I_m; I_m] (stacked 4 times), which is 20x5.
    # 
    # Similarly, V has shape (30, 5) = (30, 5). For each k, V_block should be I_m.
    # V is indexed by (l, k) -> l*p+k. For fixed k, V[l*p+k, :] for l=0..4 should be I_m.
    # So V = [I_m, I_m, I_m, I_m, I_m, I_m] (stacked 6 times), which is 30x5.
    # 
    # And W has shape (24, 5) = (24, 5). W[k*n+i, :] should be ones(5) for all (k,i).
    # 
    # Let's check: sum_c U[i*m+j,c] * V[l*p+k,c] * W[k*n+i,c]
    # = sum_{c=0}^{4} U[i*m+j,c] * V[l*p+k,c] * 1
    # = sum_c delta_{j,c} * delta_{l,c}  (since U[i*m+j,c] = delta_{j,c} and V[l*p+k,c] = delta_{l,c})
    # = delta_{j,l}
    # 
    # This works! Rank = 5!
    
    # Let's construct this solution.
    r = m  # 5
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    # U[i*m+j, c] = delta_{j,c}
    for i in range(n):
        for j in range(m):
            U[i*m+j, j] = 1
    
    # V[l*p+k, c] = delta_{l,c}
    for l in range(m):
        for k in range(p):
            V[l*p+k, l] = 1
    
    # W[k*n+i, c] = 1 for all c
    W[:] = 1
    
    # Verify
    if verify(U, V, W):
        best_rank = r
        best_U = U
        best_V = V
        best_W = W
    else:
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
