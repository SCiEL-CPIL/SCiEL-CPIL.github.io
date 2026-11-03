"""Matrix multiplication 4x4x6 via mixed Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

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

    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]  # 24 x 77
    
    # 1. 4x4x4 Strassen (columns 0-3)
    # U_444[i*4+j, 7t+s] = U_s[2I+J, t] * U_s[2i0+j0, s]
    # i = 2I + i0, j = 2J + j0
    # t, s in 0..6
    
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val
                        
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val
                        
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # 2. 4x4x2 Strassen-block (columns 4-5)
    # 4 products:
    # P1: A00 * B00 -> C00 (I=0, J=0, K=4)
    # P2: A01 * B10 -> C01 (I=0, J=1, K=4) -- Wait, B10 is J=1, K=4?
    # B is 4x6. B_JK. J in 0..3, K in 0..5.
    # B00 is J=0, K=4? No, B is 4x6.
    # Let's stick to the block decomposition of B.
    # B is 4x6. We split it into two 4x2 parts: B_left (cols 0-3) and B_right (cols 4-5).
    # B_right has columns 4,5.
    # We want to compute C_right = A * B_right.
    # A is 4x4. B_right is 4x2.
    # Partition A into 2x2 blocks A_IJ (I,J in 0..1).
    # Partition B_right into 2x2 blocks? B_right is 4x2.
    # We can partition B_right into two 2x2 blocks vertically: B_0 (rows 0-1, cols 4-5) and B_1 (rows 2-3, cols 4-5).
    # Let's call them B_top and B_bot.
    # B_top is 2x2. B_bot is 2x2.
    # C_right = A * B_right = [A_00 A_01; A_10 A_11] * [B_top; B_bot]
    # = [A_00 B_top + A_01 B_bot; A_10 B_top + A_11 B_bot]
    # So we need 4 products:
    # M1 = A_00 B_top
    # M2 = A_01 B_bot
    # M3 = A_10 B_top
    # M4 = A_11 B_bot
    
    # Indices:
    # A_IJ: I,J in 0..1. Rows 2I..2I+1, Cols 2J..2J+1.
    # B_top: Rows 0..1, Cols 4..5.
    # B_bot: Rows 2..3, Cols 4..5.
    # C_IK: I in 0..1, K in 0..1 (local to right part, global 4..5).
    # C_00 (global I=0, K=4) gets M1.
    # C_01 (global I=0, K=5) gets M1.
    # Wait, M1 is 2x2. It contributes to C[0..1, 4..5].
    # M2 contributes to C[0..1, 4..5] as well?
    # C_top = A_00 B_top + A_01 B_bot.
    # C_bot = A_10 B_top + A_11 B_bot.
    
    # So:
    # M1 = A_00 B_top -> adds to C[0..1, 4..5]
    # M2 = A_01 B_bot -> adds to C[0..1, 4..5]
    # M3 = A_10 B_top -> adds to C[2..3, 4..5]
    # M4 = A_11 B_bot -> adds to C[2..3, 4..5]
    
    # We use 2x2 Strassen for each.
    # M1: A_00 (I=0, J=0) * B_top (J=0, K=4)
    # M2: A_01 (I=0, J=1) * B_bot (J=1, K=4)
    # M3: A_10 (I=1, J=0) * B_top (J=0, K=4)
    # M4: A_11 (I=1, J=1) * B_bot (J=1, K=4)
    
    # Note: B_top is B[0..1, 4..5]. B_bot is B[2..3, 4..5].
    # In V matrix, index is j*6+k.
    # B_top: j in 0..1, k in 4..5.
    # B_bot: j in 2..3, k in 4..5.
    
    # For each product, we add 7 columns.
    # Total 28 columns, offset by 49.
    
    products = [
        (0, 0, 0), # I, J, K_offset (0 for top, 1 for bot? No, K is always 4..5)
        (0, 1, 1), # I=0, J=1, Bot
        (1, 0, 0), # I=1, J=0, Top
        (1, 1, 1), # I=1, J=1, Bot
    ]
    # Wait, the K index in B is 4 or 5.
    # B_top uses j=0,1. B_bot uses j=2,3.
    # The "K" in the 2x2 Strassen for B is the local column index 0 or 1.
    # Global k = 4 + k_local.
    
    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7
        
        # A_IJ
        # i = 2I + i0, j = 2J + j0
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_offset + s] += val
                        
        # B part
        # If is_bot, j_global = 2 + j0. Else j_global = j0.
        # k_global = 4 + k0.
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_offset + s] += val
                        
        # C part
        # I determines row block of C.
        # k_global = 4 + k0.
        # i_global = 2 * I + i0.
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_offset + s] += val

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        # Fallback
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
