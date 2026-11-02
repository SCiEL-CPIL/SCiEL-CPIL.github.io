"""Matrix multiplication 2x2 x 2x16 via shared Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W):
    """Verify that U, V, W form a valid decomposition of the 2x2 x 2x16 matmul tensor.
    
    The tensor T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'}
    So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} * delta_{k,k'}
    """
    n, m, p = 2, 2, 16
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1 if (j == j2 and k == k2) else 0
                            s = 0
                            for r in range(rank):
                                s += U[i*m + j, r] * V[j2*p + k, r] * W[k2*n + i2, r]
                            if s != expected:
                                return False
    return True

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_56():
    """8 independent Strassen blocks, rank 56."""
    n, m, p = 2, 2, 16
    U_core, V_core, W_core = _strassen_2x2()
    num_pairs = p // 2
    rank = 7 * num_pairs
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    for t in range(num_pairs):
        col_offset = t * 7
        for r in range(7):
            U[:, col_offset + r] = U_core[:, r]
            V[2*t, col_offset + r] = V_core[0, r]
            V[2*t + 1, col_offset + r] = V_core[1, r]
            V[16 + 2*t, col_offset + r] = V_core[2, r]
            V[16 + 2*t + 1, col_offset + r] = V_core[3, r]
            W[4*t, col_offset + r] = W_core[0, r]
            W[4*t + 2, col_offset + r] = W_core[1, r]
            W[4*t + 1, col_offset + r] = W_core[2, r]
            W[4*t + 3, col_offset + r] = W_core[3, r]
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    """
    # The 2x2 x 2x16 matmul tensor decomposes as 8 independent 2x2 x 2x2
    # sub-problems (one per pair of k-values), each solvable with Strassen
    # at rank 7. Total rank = 56.
    U, V, W = _construct_rank_56()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
