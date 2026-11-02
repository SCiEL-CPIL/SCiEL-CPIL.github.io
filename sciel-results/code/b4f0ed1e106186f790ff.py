import numpy as np
import time

def construct():
    """
    Rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Structure: 4 independent 2x3x2 blocks, each using Strassen(7) + 4 rank-1 = 11.
    Total rank = 4 * 11 = 44.
    """
    n, m, p = 2, 3, 8
    rank_per_block = 11
    num_blocks = p // 2  # 4
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        k0 = 2 * blk      # first output column index
        k1 = 2 * blk + 1  # second output column index

        # W indices: W[k*n + i]
        # For k=k0: W[k0*2 + 0] = W[4*blk], W[k0*2 + 1] = W[4*blk+1]
        # For k=k1: W[k1*2 + 0] = W[4*blk+2], W[k1*2 + 1] = W[4*blk+3]

        # --- Strassen part for A[:,0:2] * B[0:2,:] (2x2 * 2x2) ---
        # m1 = (A00+A11)(B00+B11)
        U[0, base + 0] = 1  # A[0,0]
        U[4, base + 0] = 1  # A[1,1]
        V[k0, base + 0] = 1        # B[0,k0]
        V[p + k1, base + 0] = 1    # B[1,k1]
        W[4*blk + 0, base + 0] = 1   # C[0,k0]
        W[4*blk + 3, base + 0] = 1   # C[1,k1]

        # m2 = (A10+A11)B00
        U[3, base + 1] = 1  # A[1,0]
        U[4, base + 1] = 1  # A[1,1]
        V[k0, base + 1] = 1        # B[0,k0]
        W[4*blk + 1, base + 1] = 1   # C[1,k0]
        W[4*blk + 3, base + 1] = -1  # C[1,k1]

        # m3 = A00(B01-B11)
        U[0, base + 2] = 1  # A[0,0]
        V[k1, base + 2] = 1        # B[0,k1]
        V[p + k1, base + 2] = -1   # B[1,k1]
        W[4*blk + 2, base + 2] = 1   # C[0,k1]
        W[4*blk + 3, base + 2] = 1   # C[1,k1]

        # m4 = A11(B10-B00)
        U[4, base + 3] = 1  # A[1,1]
        V[p + k0, base + 3] = 1    # B[1,k0]
        V[k0, base + 3] = -1       # B[0,k0]
        W[4*blk + 0, base + 3] = 1   # C[0,k0]
        W[4*blk + 1, base + 3] = 1   # C[1,k0]

        # m5 = (A00+A01)B11
        U[0, base + 4] = 1  # A[0,0]
        U[1, base + 4] = 1  # A[0,1]
        V[p + k1, base + 4] = 1   # B[1,k1]
        W[4*blk + 0, base + 4] = -1  # C[0,k0]
        W[4*blk + 2, base + 4] = 1   # C[0,k1]

        # m6 = (A10-A00)(B00+B01)
        U[3, base + 5] = 1   # A[1,0]
        U[0, base + 5] = -1  # A[0,0]
        V[k0, base + 5] = 1        # B[0,k0]
        V[k1, base + 5] = 1        # B[0,k1]
        W[4*blk + 3, base + 5] = 1  # C[1,k1]

        # m7 = (A01-A11)(B10+B11)
        U[1, base + 6] = 1   # A[0,1]
        U[4, base + 6] = -1  # A[1,1]
        V[p + k0, base + 6] = 1    # B[1,k0]
        V[p + k1, base + 6] = 1    # B[1,k1]
        W[4*blk + 0, base + 6] = 1  # C[0,k0]

        # --- Extra part: A[:,2] * B[2,:] (2x1 * 1x2) ---
        # C[0,k0] += A[0,2]*B[2,k0]
        U[2, base + 7] = 1
        V[2*p + k0, base + 7] = 1
        W[4*blk + 0, base + 7] = 1

        # C[0,k1] += A[0,2]*B[2,k1]
        U[2, base + 8] = 1
        V[2*p + k1, base + 8] = 1
        W[4*blk + 2, base + 8] = 1

        # C[1,k0] += A[1,2]*B[2,k0]
        U[5, base + 9] = 1
        V[2*p + k0, base + 9] = 1
        W[4*blk + 1, base + 9] = 1

        # C[1,k1] += A[1,2]*B[2,k1]
        U[5, base + 10] = 1
        V[2*p + k1, base + 10] = 1
        W[4*blk + 3, base + 10] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
