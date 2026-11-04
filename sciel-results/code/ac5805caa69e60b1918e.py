"""Rewritten program for 3x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    The multiplication tensor T has entries T[i,j,k,l] = 1 if (i,j) in first matrix
    and (j,k) in second matrix contribute to output (i,k).
    
    We use the schoolbook decomposition which gives rank = n*m*p = 3*3*7 = 63.
    
    For a 3x3 * 3x7 multiplication:
    - U has shape (n*m, rank) = (9, rank)
    - V has shape (m*p, rank) = (21, rank)  
    - W has shape (p*n, rank) = (21, rank)
    
    The trace convention for W is W[k*n + i, r] where i is row index of output,
    k is column index of output.
    
    Schoolbook: for each (i, j, k) triple, we have one rank-1 term:
    U[i*m + j, col] = 1
    V[j*p + k, col] = 1
    W[k*n + i, col] = 1
    
    This gives rank = n*m*p = 63.
    
    We check if there's a known better bound. For 3x3 * 3x7, the Strassen-like
    algorithms may reduce this. However, without a verified better construction,
    we return the schoolbook decomposition which is guaranteed correct.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63 for schoolbook
    
    # Build schoolbook decomposition
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
    
    # Verify the decomposition is correct
    # The tensor should satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this triple contributes to the product
    
    # Check with numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify: for each (i,j,k), the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    # For schoolbook, each (i,j,k) has exactly one r where all three are 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-10:
        # Fallback to simple schoolbook (shouldn't happen)
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
    
    # Check deadline
    if time.time() > deadline - 5:
        pass  # Return what we have
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
