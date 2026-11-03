import time
import numpy as np

DIMENSIONS = [4, 4, 6]

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

    # Strassen products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # Define (U_block, V_block, W_block) for each M
    # U_block: list of (i, j, coeff)
    # V_block: list of (j, k, coeff)
    # W_block: list of (i, k, coeff) -> W index is k*n+i
    
    prods = [
        # M1
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        # M2
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        # M3
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        # M4
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        # M5
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        # M6
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        # M7
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        # For each term in the sum for this M, we add to the column m_idx
        # But wait, U, V, W are vectors for the rank-1 term.
        # The rank-1 term is U_vec * V_vec * W_vec.
        # U_vec[i*2+j] = sum of coeffs for (i,j) in Ub
        # V_vec[j*2+k] = sum of coeffs for (j,k) in Vb
        # W_vec[k*2+i] = sum of coeffs for (i,k) in Wb
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Block decomposition
    # A is 4x4, B is 4x6, C is 4x6
    # Partition into 2x2 blocks
    # A: 2x2 blocks of 2x2
    # B: 2x3 blocks of 2x2
    # C: 2x3 blocks of 2x2
    
    # We need to compute C_IK = sum_J A_IJ * B_JK
    # I in {0,1}, K in {0,1,2}, J in {0,1}
    # There are 2 * 3 * 2 = 12 block multiplications.
    # Each is a 2x2 * 2x2 multiplication.
    # We use Strassen for each, rank 7.
    # Total rank = 12 * 7 = 84.
    
    rank = 84
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 84
    
    # Get Strassen factors for 2x2x2
    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    # Loop over block multiplications
    # I: 0,1 (row of A blocks)
    # J: 0,1 (col of A blocks / row of B blocks)
    # K: 0,1,2 (col of B blocks)
    
    col_offset = 0
    for I in range(2):
        for K in range(3):
            for J in range(2):
                # This block product A_IJ * B_JK contributes to C_IK
                # Add 7 Strassen terms
                
                # Map local 2x2 indices to global indices
                # A local (ii, jj) -> global i = 2*I + ii, j = 2*J + jj
                # B local (jj, kk) -> global j = 2*J + jj, k = 2*K + kk
                # C local (ii, kk) -> global i = 2*I + ii, k = 2*K + kk
                # W global index: k*n + i
                
                for s in range(7):
                    # U_s[ii*2+jj, s]
                    # V_s[jj*2+kk, s]
                    # W_s[kk*2+ii, s]
                    
                    for ii in range(2):
                        for jj in range(2):
                            u_val = U_s[ii * 2 + jj][s]
                            if u_val != 0:
                                global_i = 2 * I + ii
                                global_j = 2 * J + jj
                                U[global_i * m + global_j][col_offset + s] += u_val
                                
                    for jj in range(2):
                        for kk in range(2):
                            v_val = V_s[jj * 2 + kk][s]
                            if v_val != 0:
                                global_j = 2 * J + jj
                                global_k = 2 * K + kk
                                V[global_j * p + global_k][col_offset + s] += v_val
                                
                    for ii in range(2):
                        for kk in range(2):
                            w_val = W_s[kk * 2 + ii][s]
                            if w_val != 0:
                                global_i = 2 * I + ii
                                global_k = 2 * K + kk
                                W[global_k * n + global_i][col_offset + s] += w_val
                                
                col_offset += 7

    # Verify
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    
    if err > 1e-6:
        # Fallback to schoolbook
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
