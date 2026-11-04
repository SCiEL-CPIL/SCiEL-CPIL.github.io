"""Matrix multiplication 4x4x6 via three 4x4x2 Strassen blocks."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np


def _strassen_2x2_factors():
    """
    Return (U, V, W) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U shape (4, 7): rows indexed by i*2+j
    V shape (4, 7): rows indexed by j*2+k
    W shape (4, 7): rows indexed by k*2+i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)],
         [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)],
         [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)],
         [(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)],
         [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)]),
    ]

    for s, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][s] += c
        for (j, k, c) in Vb:
            V[j * p + k][s] += c
        for (i, k, c) in Wb:
            W[k * n + i][s] += c

    return U, V, W


def _rank28_4x4x2_factors():
    """
    Return (U, V, W) for the rank-28 decomposition of 4x4x2 matrix multiplication.
    Uses 4 Strassen 2x2x2 blocks.
    U shape (16, 28): rows indexed by i*4+j
    V shape (8, 28): rows indexed by j*2+k (j in 0..3, k in 0..1) -> j*2+k
    W shape (8, 28): rows indexed by k*4+i (k in 0..1, i in 0..3) -> k*4+i
    """
    U_s, V_s, W_s = _strassen_2x2_factors()
    
    U = [[0.0] * 28 for _ in range(16)]
    V = [[0.0] * 28 for _ in range(8)]
    W = [[0.0] * 28 for _ in range(8)]
    
    # 4 products:
    # 1. A_00 * B_top -> C_top (I=0, J=0, K=0)
    # 2. A_01 * B_bot -> C_top (I=0, J=1, K=1)
    # 3. A_10 * B_top -> C_bot (I=1, J=0, K=0)
    # 4. A_11 * B_bot -> C_bot (I=1, J=1, K=1)
    
    products = [
        (0, 0, 0), # I, J, K_local (0 for top, 1 for bot)
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    
    for p_idx, (I, J, K_loc) in enumerate(products):
        offset = p_idx * 7
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    U[i * 4 + j][offset + s] = U_s[2 * i0 + j0][s]
                    
        for j0 in range(2):
            for k0 in range(2):
                j = 2 * K_loc + j0
                k = k0
                for s in range(7):
                    V[j * 2 + k][offset + s] = V_s[2 * j0 + k0][s]
                    
        for i0 in range(2):
            for k0 in range(2):
                i = 2 * I + i0
                k = k0
                for s in range(7):
                    W[k * 4 + i][offset + s] = W_s[2 * k0 + i0][s]
                    
    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Structural change: Split 4x4x6 into three 4x4x2 blocks.
    # Each 4x4x2 block is decomposed using rank-28 (4 Strassen blocks).
    # Total rank = 3 * 28 = 84.
    
    U_block, V_block, W_block = _rank28_4x4x2_factors()
    
    rank = 84
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Block 0: k in 0..1
    # Block 1: k in 2..3
    # Block 2: k in 4..5
    
    for b in range(3):
        offset = b * 28
        for i in range(16):
            U[i][offset + (i % 28)] = 0 # Clearing not needed, initialized to 0
            for c in range(28):
                U[i][offset + c] = U_block[i][c]
                
        for j in range(8):
            for c in range(28):
                # V_block index: j*2 + k_local. 
                # Global k = b*2 + k_local.
                # Global V row: j_global * 6 + k_global.
                # j_global is the same j (0..3).
                # k_global = b*2 + (j*2+c % 2? No, V_block row is j*2+k).
                # V_block row r corresponds to j=r//2, k=r%2.
                j_local = j // 2
                k_local = j % 2
                k_global = b * 2 + k_local
                j_global = j_local
                for c in range(28):
                    V[j_global * 6 + k_global][offset + c] = V_block[j][c]
                    
        for k in range(8):
            for c in range(28):
                # W_block index: k*4 + i.
                # k_local = k // 4. i = k % 4.
                k_local = k // 4
                i = k % 4
                k_global = b * 2 + k_local
                for c in range(28):
                    W[k_global * 4 + i][offset + c] = W_block[k][c]

    err = _verify(U, V, W, n, m, p, rank)
    if err > 1e-6:
        # Fallback to Schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
