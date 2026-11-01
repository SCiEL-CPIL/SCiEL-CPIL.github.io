import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (u_vec, v_vec, w_vec) where u_vec, v_vec, w_vec are length 4.
    u_vec corresponds to A entries [a11, a12, a21, a22]
    v_vec corresponds to B entries [b11, b12, b21, b22]
    w_vec corresponds to C entries [c11, c12, c21, c22]
    """
    terms = []
    
    # M1 = (a11+a22)(b11+b22)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    # c11 += M1, c22 += M1
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (a21+a22)b11
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    # c21 += M2
    w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    
    # M3 = a11(b22-b11)
    u3 = [1, 0, 0, 0]
    v3 = [-1, 0, 0, 1]
    # c12 += M3, c22 += M3
    w3 = [0, 1, 0, 1]
    terms.append((u3, v3, w3))
    
    # M4 = a12(b21+b22)
    u4 = [0, 1, 0, 0]
    v4 = [0, 0, 1, 1]
    # c11 += M4, c21 += M4
    w4 = [1, 0, 1, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (a11-a12)(b11+b21)
    u5 = [1, -1, 0, 0]
    v5 = [1, 0, 1, 0]
    # c11 -= M5, c12 += M5
    w5 = [-1, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (a21-a11)b12
    u6 = [-1, 0, 1, 0]
    v6 = [0, 1, 0, 0]
    # c22 += M6
    w6 = [0, 0, 0, 1]
    terms.append((u6, v6, w6))
    
    # M7 = (a22-a21)(b12-b11)
    u7 = [0, 0, -1, 1]
    v7 = [-1, 1, 0, 0]
    # c11 += M7
    w7 = [1, 0, 0, 0]
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # Total rank is 56
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_terms = _strassen_2x2_terms()
    
    col = 0
    
    # Helper to add a Strassen term for a specific 2x2x2 block
    # a_offset: index in U for the first element of the 2x2 A block
    # b_offset: index in V for the first element of the 2x2 B block
    # c_offset: index in W for the first element of the 2x2 C block
    # a_stride: stride for A elements (always 1 in our flat index? No, A is 2x4)
    # We need to map local 2x2 indices to global indices.
    
    def add_strassen_block(a_base, b_base, c_base, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        for u, v, w in strassen_terms:
            for li in range(4):
                # Local index li in 2x2 block: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
                lr, lc = divmod(li, 2)
                
                # Global indices
                ga = a_base + a_rows[lr] * (a_cols[1]-a_cols[0]+1) + a_cols[lc] # This logic is fragile
                # Better to pass explicit mapping or calculate based on known structure.
                
            # Simplified: We will hardcode the mappings for the 8 blocks.
            pass

    # Block 1: A_L (cols 0,1) * B_L1 (rows 0,1, cols 0,1) -> C_L (rows 0,1, cols 0,1)
    # A_L elements: U[0], U[1], U[4], U[5] (indices i*m+j)
    # B_L1 elements: V[0], V[1], V[4], V[5] (indices j*p+k)
    # C_L elements: W[0], W[1], W[2], W[3] (indices k*n+i)
    
    # Mapping for A_L (2x2):
    # (0,0) -> U[0]
    # (0,1) -> U[1]
    # (1,0) -> U[4]
    # (1,1) -> U[5]
    a_map_1 = [0, 1, 4, 5]
    
    # Mapping for B_L1 (2x2):
    # (0,0) -> V[0]
    # (0,1) -> V[1]
    # (1,0) -> V[4]
    # (1,1) -> V[5]
    b_map_1 = [0, 1, 4, 5]
    
    # Mapping for C_L (2x2, cols 0,1):
    # (0,0) -> W[0]
    # (0,1) -> W[1]
    # (1,0) -> W[2]
    # (1,1) -> W[3]
    c_map_1 = [0, 1, 2, 3]
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_1[li], col] += u[li]
            V[b_map_1[li], col] += v[li]
            W[c_map_1[li], col] += w[li]
        col += 1

    # Block 2: A_L (cols 0,1) * B_L1 (rows 0,1, cols 2,3) -> C_L (rows 0,1, cols 2,3)
    # A_L same.
    # B_L1 cols 2,3:
    # (0,0) -> V[2]
    # (0,1) -> V[3]
    # (1,0) -> V[6]
    # (1,1) -> V[7]
    b_map_2 = [2, 3, 6, 7]
    
    # C_L cols 2,3:
    # (0,0) -> W[4]
    # (0,1) -> W[5]
    # (1,0) -> W[6]
    # (1,1) -> W[7]
    c_map_2 = [4, 5, 6, 7]
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_1[li], col] += u[li]
            V[b_map_2[li], col] += v[li]
            W[c_map_2[li], col] += w[li]
        col += 1

    # Block 3: A_R (cols 2,3) * B_L2 (rows 2,3, cols 0,1) -> C_L (rows 0,1, cols 0,1)
    # A_R elements:
    # (0,0) -> U[2]
    # (0,1) -> U[3]
    # (1,0) -> U[6]
    # (1,1) -> U[7]
    a_map_3 = [2, 3, 6, 7]
    
    # B_L2 rows 2,3, cols 0,1:
    # (0,0) -> V[8]
    # (0,1) -> V[9]
    # (1,0) -> V[12]
    # (1,1) -> V[13]
    b_map_3 = [8, 9, 12, 13]
    
    # C_L cols 0,1 (same as block 1)
    c_map_3 = c_map_1
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_3[li], col] += u[li]
            V[b_map_3[li], col] += v[li]
            W[c_map_3[li], col] += w[li]
        col += 1

    # Block 4: A_R (cols 2,3) * B_L2 (rows 2,3, cols 2,3) -> C_L (rows 0,1, cols 2,3)
    # A_R same.
    # B_L2 rows 2,3, cols 2,3:
    # (0,0) -> V[10]
    # (0,1) -> V[11]
    # (1,0) -> V[14]
    # (1,1) -> V[15]
    b_map_4 = [10, 11, 14, 15]
    
    # C_L cols 2,3 (same as block 2)
    c_map_4 = c_map_2
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_3[li], col] += u[li]
            V[b_map_4[li], col] += v[li]
            W[c_map_4[li], col] += w[li]
        col += 1

    # Block 5: A_L (cols 0,1) * B_R1 (rows 0,1, cols 0,1) -> C_R (rows 0,1, cols 0,1)
    # A_L same.
    # B_R1 is B[:, 4:8] rows 0,1.
    # (0,0) -> V[16]
    # (0,1) -> V[17]
    # (1,0) -> V[20]
    # (1,1) -> V[21]
    b_map_5 = [16, 17, 20, 21]
    
    # C_R cols 0,1 (global cols 4,5):
    # (0,0) -> W[8]
    # (0,1) -> W[9]
    # (1,0) -> W[10]
    # (1,1) -> W[11]
    c_map_5 = [8, 9, 10, 11]
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_1[li], col] += u[li]
            V[b_map_5[li], col] += v[li]
            W[c_map_5[li], col] += w[li]
        col += 1

    # Block 6: A_L (cols 0,1) * B_R1 (rows 0,1, cols 2,3) -> C_R (rows 0,1, cols 2,3)
    # A_L same.
    # B_R1 rows 0,1, cols 2,3:
    # (0,0) -> V[18]
    # (0,1) -> V[19]
    # (1,0) -> V[22]
    # (1,1) -> V[23]
    b_map_6 = [18, 19, 22, 23]
    
    # C_R cols 2,3 (global cols 6,7):
    # (0,0) -> W[12]
    # (0,1) -> W[13]
    # (1,0) -> W[14]
    # (1,1) -> W[15]
    c_map_6 = [12, 13, 14, 15]
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_1[li], col] += u[li]
            V[b_map_6[li], col] += v[li]
            W[c_map_6[li], col] += w[li]
        col += 1

    # Block 7: A_R (cols 2,3) * B_R2 (rows 2,3, cols 0,1) -> C_R (rows 0,1, cols 0,1)
    # A_R same.
    # B_R2 rows 2,3, cols 0,1:
    # (0,0) -> V[24]
    # (0,1) -> V[25]
    # (1,0) -> V[28]
    # (1,1) -> V[29]
    b_map_7 = [24, 25, 28, 29]
    
    # C_R cols 0,1 (same as block 5)
    c_map_7 = c_map_5
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_3[li], col] += u[li]
            V[b_map_7[li], col] += v[li]
            W[c_map_7[li], col] += w[li]
        col += 1

    # Block 8: A_R (cols 2,3) * B_R2 (rows 2,3, cols 2,3) -> C_R (rows 0,1, cols 2,3)
    # A_R same.
    # B_R2 rows 2,3, cols 2,3:
    # (0,0) -> V[26]
    # (0,1) -> V[27]
    # (1,0) -> V[30]
    # (1,1) -> V[31]
    b_map_8 = [26, 27, 30, 31]
    
    # C_R cols 2,3 (same as block 6)
    c_map_8 = c_map_6
    
    for u, v, w in strassen_terms:
        for li in range(4):
            U[a_map_3[li], col] += u[li]
            V[b_map_8[li], col] += v[li]
            W[c_map_8[li], col] += w[li]
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
