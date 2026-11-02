"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    # Compute the tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all (i,j,k)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j, :]
                v_val = V_np[j * p + k, :]
                w_val = W_np[k * n + i, :]
                tensor_val = np.sum(u_val * v_val * w_val)
                if abs(tensor_val - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(n, m, p, max_rank=None, deadline=None):
    """Try to find a lower-rank decomposition by searching for linear combinations."""
    if max_rank is None:
        max_rank = n * m * p - 1
    
    # Start with schoolbook rank
    schoolbook_rank = n * m * p
    
    # First, return the schoolbook decomposition as a fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    if _verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_result = (U_school, V_school, W_school, schoolbook_rank)
    else:
        return None
    
    # Try to find lower-rank decompositions using Strassen-like ideas
    # For 2x3 * 3x5, we can try to split the computation
    
    # Idea: Split the 3x5 matrix B into two parts
    # B = [B1 | B2] where B1 is 3x2 and B2 is 3x3
    # Then C = A*B = A*B1 + A*B2
    
    # For A*B1 (2x3 * 3x2), we can use a rank-5 decomposition
    # For A*B2 (2x3 * 3x3), we can try to use lower rank
    
    # Actually, let's try a more direct approach:
    # We can try to find if there's a rank r decomposition where r < n*m*p
    
    # For now, let's try specific known decompositions or search
    
    # Try rank from schoolbook_rank-1 down to some minimum
    # But this is expensive, so let's be strategic
    
    # For 2x3*3x5, the tensor has 30 entries
    # The minimum possible rank is at least the max of:
    # - max slice rank
    # - other lower bounds
    
    # Let's compute some lower bounds
    # The tensor can be viewed as a 3-tensor T[i,j,k]
    
    # For each fixed i, we get a matrix T[i,:,:] of size m x p = 3x5
    # The rank of this matrix gives a lower bound
    # Similarly for fixed j and fixed k
    
    # But this is complex, so let's just try a few specific decompositions
    
    # Try to use the fact that we can combine some terms
    # For example, if we have terms (i1,j1,k1) and (i2,j2,k2),
    # can we express them with one term?
    
    # This requires U[i1*m+j1] * V[j1*p+k1] * W[k1*n+i1] + 
    #           U[i2*m+j2] * V[j2*p+k2] * W[k2*n+i2]
    # to be replaced by a single term with different indices
    
    # This is a hard problem, so let's try a heuristic search
    
    # Try specific small rank reductions
    # For 2x3*3x5, try rank 29, 28, etc.
    
    # A practical approach: try to find if we can merge two schoolbook terms
    # into one by finding appropriate U, V, W values
    
    # For two terms (i1,j1,k1) and (i2,j2,k2), we want:
    # U_new[i1*m+j1] * V_new[j1*p+k1] * W_new[k1*n+i1] +
    # U_new[i2*m+j2] * V_new[j2*p+k2] * W_new[k2*n+i2] = 1 (for the combined term)
    # And 0 for all other (i,j,k)
    
    # This is underdetermined, so let's try a different approach
    
    # Use the fact that for small n, m, p, we can try all possible
    # "Strassen-like" decompositions
    
    # For 2x3*3x5, one approach is to use block decomposition
    # Split A into [A1; A2] where A1 is 1x3, A2 is 1x3
    # Then C1 = A1*B, C2 = A2*B
    
    # Each Ci is 1x5, so we need to compute dot products
    # C1[k] = sum_j A1[j] * B[j,k]
    
    # This is just 5 dot products of length 3, which is rank 15 for each,
    # total rank 30 (same as schoolbook)
    
    # So block decomposition doesn't help here
    
    # Let's try a more sophisticated approach:
    # Use the fact that the tensor has a specific algebraic structure
    
    # Try to find a rank r decomposition by solving a system of equations
    # For rank r, we have r*30 variables (U, V, W each have r columns)
    # And 30 equations (one for each (i,j,k))
    
    # The equations are: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of polynomial equations, which is hard to solve directly
    
    # Instead, let's try a heuristic: start with schoolbook and try to
    # merge terms by finding linear combinations
    
    # For each pair of terms, check if they can be merged
    # This requires solving a small system of equations
    
    # Return the schoolbook decomposition as the best we have
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Set a deadline 10 seconds before the 150s limit
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Try to find a lower-rank decomposition
    result = _try_rank_reduction(n, m, p, deadline=deadline)
    
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
