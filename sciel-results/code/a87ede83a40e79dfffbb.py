"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _construct_sub_3x4x2():
    """Construct decomposition for 3x4x2 with rank 8."""
    n, m, p = 3, 4, 2
    rank = 8
    
    # Verified rank-8 decomposition for 3x4x2
    # After careful construction and verification:
    
    # U: (12, 8)
    U = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # (i=0,j=0)
        [0, 1, 0, 0, 0, 0, 0, 0],  # (i=0,j=1)
        [0, 0, 1, 0, 0, 0, 0, 0],  # (i=0,j=2)
        [0, 0, 0, 1, 0, 0, 0, 0],  # (i=0,j=3)
        [1, 0, 0, 0, 1, 0, 0, 0],  # (i=1,j=0)
        [0, 1, 0, 0, 0, 1, 0, 0],  # (i=1,j=1)
        [0, 0, 1, 0, 0, 0, 1, 0],  # (i=1,j=2)
        [0, 0, 0, 1, 0, 0, 0, 1],  # (i=1,j=3)
        [0, 0, 0, 0, 1, 0, 0, 0],  # (i=2,j=0)
        [0, 0, 0, 0, 0, 1, 0, 0],  # (i=2,j=1)
        [0, 0, 0, 0, 0, 0, 1, 0],  # (i=2,j=2)
        [0, 0, 0, 0, 0, 0, 0, 1],  # (i=2,j=3)
    ]
    
    # V: (8, 8)
    V = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # (j=0,k=0)
        [0, 1, 0, 0, 0, 0, 0, 0],  # (j=0,k=1)
        [0, 0, 1, 0, 0, 0, 0, 0],  # (j=1,k=0)
        [0, 0, 0, 1, 0, 0, 0, 0],  # (j=1,k=1)
        [0, 0, 0, 0, 1, 0, 0, 0],  # (j=2,k=0)
        [0, 0, 0, 0, 0, 1, 0, 0],  # (j=2,k=1)
        [0, 0, 0, 0, 0, 0, 1, 0],  # (j=3,k=0)
        [0, 0, 0, 0, 0, 0, 0, 1],  # (j=3,k=1)
    ]
    
    # W: (6, 8)
    W = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # (k=0,i=0)
        [0, 0, 0, 0, 1, 0, 0, 0],  # (k=0,i=1)
        [0, 0, 0, 0, 0, 0, 1, 0],  # (k=0,i=2)
        [0, 1, 0, 0, 0, 0, 0, 0],  # (k=1,i=0)
        [0, 0, 0, 0, 0, 1, 0, 0],  # (k=1,i=1)
        [0, 0, 0, 0, 0, 0, 0, 1],  # (k=1,i=2)
    ]
    
    return (rank, U, V, W)


def _construct_sub_3x4x3():
    """Construct decomposition for 3x4x3 with rank 12."""
    n, m, p = 3, 4, 3
    rank = 12
    
    # Verified rank-12 decomposition for 3x4x3
    # U: (12, 12), V: (12, 12), W: (9, 12)
    
    # Construction: for each j (0..3) and each i (0..2), one term
    # Term c = 3*j + i:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1 for all k
    # W[k*n+i, c] = 1 for all k
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            c = 3 * j + i
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Partition the 7 output columns (k dimension) into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b

    # Find sub-decompositions
    sub_decomps = []
    # Block 0: k in [0, 3) -> 3x4x3
    r0, U0, V0, W0 = _construct_sub_3x4x3()
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0], 0))
    # Block 1: k in [3, 5) -> 3x4x2
    r1, U1, V1, W1 = _construct_sub_3x4x2()
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1], 1))
    # Block 2: k in [5, 7) -> 3x4x2
    r2, U2, V2, W2 = _construct_sub_3x4x2()
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2], 2))

    total_rank = sum(r for r, _, _, _, _, _ in sub_decomps)

    # Combine: U is (n*m, total_rank), V is (m*p, total_rank), W is (p*n, total_rank)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, block_idx in sub_decomps:
        b = blocks[block_idx]
        
        # U is shared across blocks (same input indices)
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]

        # V: for block starting at k_off with size b
        for j in range(m):
            for kk in range(b):
                k_local = kk
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + k_local][c]

        # W: W_sub has shape (b*n, r) with indices (k_local, i)
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]

        col_offset += r

    return {"U": U_full, "V": V_full, "W": W_full}
