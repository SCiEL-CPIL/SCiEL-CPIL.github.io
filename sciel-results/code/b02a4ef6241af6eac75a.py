import numpy as np
import random
import time
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 60
    
    # The tensor T[i,j,k] = delta[i,j,k] (identity-like structure for matrix mult)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Strategy: Use the structure of matrix multiplication to find a sparse decomposition.
    # For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor has structure: T[i,j,k] = 1 if this term contributes to C[i,k]
    # Actually for the multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k)
    # and we need the decomposition to satisfy:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Let's try to find a rank < 60 decomposition using combinatorial optimization.
    # 
    # Approach: Model as finding a set of "triples" (u_idx, v_idx, w_idx) with weights
    # such that for each (i,j,k), the sum of products equals 1.
    #
    # For a sparse decomposition, each column r defines a triple (u_r, v_r, w_r) where
    # u_r is a vector in R^(n*m), v_r in R^(m*p), w_r in R^(p*n).
    #
    # Simplest sparse case: each column has at most one non-zero in each of U, V, W.
    # Then U[i,j,r] ∈ {0,1}, V[j,k,r] ∈ {0,1}, W[k,i,r] ∈ {0,1}
    # and we need: for each (i,j,k), exactly one r has U[i,j,r]=V[j,k,r]=W[k,i,r]=1.
    #
    # This is equivalent to finding a perfect matching in a tripartite hypergraph.
    
    # Let's first try the schoolbook (rank 60) and then attempt to reduce.
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, base_rank), dtype=float)
    V_school = np.zeros((m * p, base_rank), dtype=float)
    W_school = np.zeros((p * n, base_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    assert check_decomposition(U_school, V_school, W_school, n, m, p)
    
    # Now try to find a lower rank decomposition.
    # 
    # Key insight: For matrix multiplication of n×m by m×p, the tensor rank is known to be
    # less than n*m*p for many cases. For 3×4×5, let's try to find a rank 59 or lower solution.
    #
    # We can use the fact that some combinations might be "redundant" if we allow half-integer
    # or other rational values.
    
    # Try rank 59: Can we merge two schoolbook terms?
    # If we have two columns r1, r2 that affect the same (i,j,k) pattern, we might combine them.
    
    # Let's use a different approach: start with schoolbook and try to eliminate one column
    # by adjusting the others.
    
    # For rank R = base_rank - 1 = 59:
    # We need to find U, V, W of shape (n*m, 59), (m*p, 59), (p*n, 59)
    # such that the tensor equation holds.
    
    # This is a system of 60 equations (one for each (i,j,k)) with 3*59 = 177 unknowns.
    # The system is underdetermined, so solutions exist if we can find them.
    
    # Let's try a constructive approach:
    # Take the schoolbook and try to "absorb" one column into others.
    
    # Pick the last column (i=2, j=3, k=4) to eliminate.
    # Set U[2*3+3, r] = 0, V[3*4+4, r] = 0, W[4*3+2, r] = 0 for all r
    # Then we need to adjust other entries to compensate.
    
    # Actually, let's try a simpler heuristic: use known results or try random sparse patterns.
    
    # For now, return the schoolbook as it's guaranteed correct.
    # The rank is 60.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
