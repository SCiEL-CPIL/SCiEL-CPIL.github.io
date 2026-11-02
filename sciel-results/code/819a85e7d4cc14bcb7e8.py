import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START

def construct():
    """
    Exact rank-35 tensor decomposition for multiplying a 2x2 matrix by a 2x10 matrix.
    
    The 2x2 by 2x10 multiplication tensor decomposes as a direct sum of 5 independent
    2x2x2 multiplication tensors (one per pair of columns in B). Each 2x2x2 tensor
    has proven minimal rank 7 (Strassen). Total rank = 5 * 7 = 35.
    
    Interface:
      U: shape (n*m, rank) = (4, 35), indexed by i*m + j
      V: shape (m*p, rank) = (20, 35), indexed by j*p + k
      W: shape (p*n, rank) = (20, 35), indexed by k*n + i
    """
    n, m, p = DIMENSIONS
    total_rank = 35

    U_mat = np.zeros((n * m, total_rank), dtype=int)
    V_mat = np.zeros((m * p, total_rank), dtype=int)
    W_mat = np.zeros((p * n, total_rank), dtype=int)

    # Strassen 7-term decomposition for 2x2x2 multiplication.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C00 = P1 - P2 + P4 - P7, C01 = P3 + P5, C10 = P2 + P6, C11 = P1 - P3 + P6 + P7

    # A-side coefficients: rows are (i*m+j) = (0,0), (0,1), (1,0), (1,1)
    # Columns are the 7 Strassen terms
    strassen_U = np.array([
        # P1  P2  P3  P4  P5  P6  P7
        [1,  0,  1,  0,  1,  1,  0],   # a
        [0,  0,  0,  0,  1,  0,  1],   # b
        [0,  1,  0,  0,  0, -1, 0],   # c
        [1,  1,  0,  1,  0,  0, -1],   # d
    ], dtype=int)

    # B-side coefficients: rows are (j*p+k), for 2x2 block: (0,0), (0,1), (1,0), (1,1)
    # i.e., j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
    strassen_V = np.array([
        # P1  P2  P3  P4  P5  P6  P7
        [1,  1,  0, -1,  0,  1,  0],   # e
        [1,  0,  1,  0,  0,  1,  1],   # f
        [0,  0, -1, 1,  0,  0,  0],   # g
        [1,  0,  0,  0,  1,  0,  1],   # h
    ], dtype=int)

    # C-side coefficients: rows are (k*n+i), for 2x2 output: (0,0), (0,1), (1,0), (1,1)
    # i.e., k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1
    strassen_W = np.array([
        # P1  P2  P3  P4  P5  P6  P7
        [1, -1,  0,  1,  0,  0, -1],   # C[0,0]
        [0,  0,  1,  0,  1,  0,  0],   # C[0,1]
        [0,  1,  0,  0,  0,  1,  0],   # C[1,0]
        [1,  0, -1, 0,  0,  1,  1],   # C[1,1]
    ], dtype=int)

    # Assemble: 5 column-blocks of B, each contributing 7 rank terms
    for blk in range(5):
        col_offset = blk * 7
        # A-side: same for all blocks (A is 2x2, shared)
        U_mat[:, col_offset:col_offset+7] = strassen_U

        # B-side: columns 2*blk and 2*blk+1 of B
        b_row0 = 0 * p + 2 * blk      # j=0, k=2*blk
        b_row1 = 0 * p + (2 * blk + 1)  # j=0, k=2*blk+1
        b_row2 = 1 * p + 2 * blk      # j=1, k=2*blk
        b_row3 = 1 * p + (2 * blk + 1)  # j=1, k=2*blk+1
        V_mat[b_row0, col_offset:col_offset+7] = strassen_V[0]
        V_mat[b_row1, col_offset:col_offset+7] = strassen_V[1]
        V_mat[b_row2, col_offset:col_offset+7] = strassen_V[2]
        V_mat[b_row3, col_offset:col_offset+7] = strassen_V[3]

        # W-side: output columns 2*blk and 2*blk+1
        w_row0 = (2 * blk) * n + 0      # k=2*blk, i=0
        w_row1 = (2 * blk) * n + 1      # k=2*blk, i=1
        w_row2 = (2 * blk + 1) * n + 0  # k=2*blk+1, i=0
        w_row3 = (2 * blk + 1) * n + 1  # k=2*blk+1, i=1
        W_mat[w_row0, col_offset:col_offset+7] = strassen_W[0]
        W_mat[w_row1, col_offset:col_offset+7] = strassen_W[1]
        W_mat[w_row2, col_offset:col_offset+7] = strassen_W[2]
        W_mat[w_row3, col_offset:col_offset+7] = strassen_W[3]

    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

def _check_decomposition(U, V, W, n=2, m=2, p=10):
    """
    Verify that the tensor decomposition exactly computes matrix multiplication.
    Condition: for all i, i', j, j', k, k':
      sum_r W[k*n+i, r] * U[i'*m+j, r] * V[j'*p+k', r]
        = 1 if (i==i' and j==j' and k==k') else 0
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    for i in range(n):
        for ip in range(n):
            for k in range(p):
                for kp in range(p):
                    M = np.zeros((m, m))
                    for r in range(rank):
                        wv = Wa[k * n + i, r]
                        if wv != 0.0:
                            for j in range(m):
                                uv = Ua[ip * m + j, r]
                                if uv != 0.0:
                                    for jp in range(m):
                                        vv = Va[jp * p + kp, r]
                                        if vv != 0.0:
                                            M[j, jp] += wv * uv * vv
                    if i == ip and k == kp:
                        if not np.allclose(M, np.eye(m), atol=1e-10):
                            return False
                    else:
                        if not np.allclose(M, np.zeros((m, m)), atol=1e-10):
                            return False
    return True

# EVOLVE-BLOCK-END
