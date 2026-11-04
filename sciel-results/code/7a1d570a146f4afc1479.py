"""Rewritten program: block/Strassen hybrid decomposition for M_{3,4,7}."""

import numpy as np

DIMENSIONS = [3, 4, 7]


def _strassen_m222():
    """Return (U, V, W) for 2x2*2x2, rank 7, indices:
       U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Strassen terms
    U[0, 0] = 1; U[0, 1] = 0; U[0, 2] = 1; U[0, 3] = 0; U[0, 4] = 1; U[0, 5] = 1; U[0, 6] = 1
    U[1, 0] = 0; U[1, 1] = 0; U[1, 2] = 1; U[1, 3] = 0; U[1, 4] = 0; U[1, 5] = 1; U[1, 6] = -1
    U[2, 0] = 0; U[2, 1] = 0; U[2, 2] = 0; U[2, 3] = 1; U[2, 4] = -1; U[2, 5] = 0; U[2, 6] = 0
    U[3, 0] = 1; U[3, 1] = 1; U[3, 2] = 0; U[3, 3] = 1; U[3, 4] = 0; U[3, 5] = -1; U[3, 6] = -1
    V[0, 0] = 1; V[0, 1] = -1; V[0, 2] = 0; V[0, 3] = 1; V[0, 4] = 1; V[0, 5] = 1; V[0, 6] = -1
    V[1, 0] = 1; V[1, 1] = 1; V[1, 2] = 0; V[1, 3] = 0; V[1, 4] = 1; V[1, 5] = 0; V[1, 6] = 1
    V[2, 0] = 0; V[2, 1] = 0; V[2, 2] = 1; V[2, 3] = 0; V[2, 4] = 1; V[2, 5] = 1; V[2, 6] = 0
    V[3, 0] = 1; V[3, 1] = 1; V[3, 2] = 1; V[3, 3] = 1; V[3, 4] = 0; V[3, 5] = 0; V[3, 6] = -1
    W[0, 0] = 1; W[0, 1] = 0; W[0, 2] = 0; W[0, 3] = 1; W[0, 4] = -1; W[0, 5] = 0; W[0, 6] = -1
    W[1, 0] = 0; W[1, 1] = 0; W[1, 2] = 1; W[1, 3] = 0; W[1, 4] = 1; W[1, 5] = 0; W[1, 6] = 0
    W[2, 0] = 0; W[2, 1] = 1; W[2, 2] = 0; W[2, 3] = 1; W[2, 4] = 0; W[2, 5] = 0; W[2, 6] = 0
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 0; W[3, 3] = 0; W[3, 4] = 0; W[3, 5] = 1; W[3, 6] = 1
    return U, V, W


def _m1mp_rank1(m, p):
    """Rank-1 decomposition for 1xm * mxp -> 1xp. rank = min(m,p)."""
    rank = min(m, p)
    U = np.zeros((1 * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * 1, rank))
    if m <= p:
        for c in range(m):
            U[0, c] = 1
            for k in range(p):
                V[c * p + k, c] = 1
            W[:, c] = 1
    else:
        for c in range(p):
            U[0, c] = 1
            for j in range(m):
                V[j * p + c, c] = 1
            W[c, c] = 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    total_U_rows = n * m
    total_V_rows = m * p
    total_W_rows = p * n
    
    # Estimate rank: 58
    rank = 60
    U_full = np.zeros((total_U_rows, rank))
    V_full = np.zeros((total_V_rows, rank))
    W_full = np.zeros((total_W_rows, rank))
    
    offset = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal offset
        U, V, W = _strassen_m222()
        # Map indices
        for i in range(2):
            for j in range(2):
                u_idx = (i0 + i) * m + (j0 + j)
                u_block_idx = i * 2 + j
                U_full[u_idx, offset:offset+7] = U[u_block_idx]
        for j in range(2):
            for k in range(2):
                v_idx = (j0 + j) * p + (k0 + k)
                v_block_idx = j * 2 + k
                V_full[v_idx, offset:offset+7] = V[v_block_idx]
        for k in range(2):
            for i in range(2):
                w_idx = (k0 + k) * n + (i0 + i)
                w_block_idx = k * 2 + i
                W_full[w_idx, offset:offset+7] = W[w_block_idx]
        offset += 7

    def add_m1mp(i0, j0, k0, ni, mj, pk):
        nonlocal offset
        U, V, W = _m1mp_rank1(mj, pk)
        r = U.shape[1]
        for i in range(ni):
            for j in range(mj):
                u_idx = (i0 + i) * m + (j0 + j)
                u_block_idx = i * mj + j
                U_full[u_idx, offset:offset+r] = U[u_block_idx]
        for j in range(mj):
            for k in range(pk):
                v_idx = (j0 + j) * p + (k0 + k)
                v_block_idx = j * pk + k
                V_full[v_idx, offset:offset+r] = V[v_block_idx]
        for k in range(pk):
            for i in range(ni):
                w_idx = (k0 + k) * n + (i0 + i)
                w_block_idx = k * ni + i
                W_full[w_idx, offset:offset+r] = W[w_block_idx]
        offset += r

    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        nonlocal offset
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U_full[(i0+i)*m + (j0+j), offset] = 1
                    V_full[(j0+j)*p + (k0+k), offset] = 1
                    W_full[(k0+k)*n + (i0+i), offset] = 1
                    offset += 1

    # Block 1: n=0:2, m=0:2, p=0:4 -> Strassen (2,2,2) + Strassen (2,2,2)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    
    # Block 2: n=0:2, m=0:2, p=4:7 -> Strassen (2,2,2) + Schoolbook (2,2,1)
    add_strassen(0, 0, 4)
    add_schoolbook(0, 0, 6, 2, 2, 1)
    
    # Block 3: n=2:3, m=0:2, p=0:4 -> M1MP (1,2,4)
    add_m1mp(2, 0, 0, 1, 2, 4)
    
    # Block 4: n=2:3, m=0:2, p=4:7 -> M1MP (1,2,3)
    add_m1mp(2, 0, 4, 1, 2, 3)
    
    # Block 5: n=0:2, m=2:4, p=0:4 -> Strassen (2,2,2) + Strassen (2,2,2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Block 6: n=0:2, m=2:4, p=4:7 -> Strassen (2,2,2) + Schoolbook (2,2,1)
    add_strassen(0, 2, 4)
    add_schoolbook(0, 2, 6, 2, 2, 1)
    
    # Block 7: n=2:3, m=2:4, p=0:4 -> M1MP (1,2,4)
    add_m1mp(2, 2, 0, 1, 2, 4)
    
    # Block 8: n=2:3, m=2:4, p=4:7 -> M1MP (1,2,3)
    add_m1mp(2, 2, 4, 1, 2, 3)
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
