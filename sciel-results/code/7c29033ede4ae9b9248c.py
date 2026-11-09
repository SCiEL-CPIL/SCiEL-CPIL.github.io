"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import null_space

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _build_tensor_matrix(n, m, p, U, V, W):
    """
    Build the linear system matrix for the decomposition constraints.
    Each row corresponds to an (i,j,k) triple and has 1 in the column
    corresponding to U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    We want to find if we can reduce the rank.
    """
    rank = U.shape[1]
    # Number of constraints: n*m*p
    n_constraints = n * m * p
    
    # For each (i,j,k), the constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is nonlinear in U, V, W.
    # 
    # Alternative approach: fix two of the matrices and solve for the third.
    # Or use a different strategy.
    pass

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    """
    Try to find a decomposition with target_rank by solving for one factor
    given the others, or by using a rank-revealing decomposition.
    """
    # Strategy: Use the fact that the tensor T[i,j,k] = 1 for all i,j,k
    # can be decomposed. We'll try to find if there's a lower rank decomposition.
    # 
    # One approach: Fix U and V, then for each (i,j,k), we need:
    # W[k*n+i, c] to satisfy the equation. But this is coupled across c.
    #
    # Better approach: Use the structure of matrix multiplication.
    # The tensor M_{i,j,k} = 1 for all valid i,j,k.
    # We can flatten this in different ways and look at ranks.
    
    # Create the tensor
    T = np.ones((n, m, p))
    
    # Flatten in different ways to get matrix representations
    # T_{i,j,k} -> matrix A_{i, (j,k)} of shape (n, m*p)
    A = T.reshape(n, m * p)
    # T_{j, (i,k)} -> matrix B_{j, (i,k)} of shape (m, n*p)  
    B = T.transpose(1, 0, 2).reshape(m, n * p)
    # T_{k, (i,j)} -> matrix C_{k, (i,j)} of shape (p, n*m)
    C = T.transpose(2, 0, 1).reshape(p, n * m)
    
    # The tensor rank is at least the maximum of the ranks of these flattening
    rank_A = np.linalg.matrix_rank(A, tol=1e-10)
    rank_B = np.linalg.matrix_rank(B, tol=1e-10)
    rank_C = np.linalg.matrix_rank(C, tol=1e-10)
    
    lower_bound = max(rank_A, rank_B, rank_C)
    
    # For T = ones((n,m,p)), all flattening have rank min(dimension)
    # rank_A = min(n, m*p) = min(3, 25) = 3
    # rank_B = min(m, n*p) = min(5, 15) = 5
    # rank_C = min(p, n*m) = min(5, 15) = 5
    # So lower bound is 5, but the actual tensor rank is higher.
    
    # The tensor rank of the all-ones tensor is not simply the flattening rank.
    # We need a different approach.
    
    # For now, return the lower bound as an estimate
    return lower_bound

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Return schoolbook as baseline
    return result

# EVOLVE-BLOCK-END
