import numpy as np

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # P2 = (A10 + A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # P3 = A00(B01 - B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # P4 = A11(B10 - B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # P5 = (A00 + A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _block_strassen_4x4x8():
    """
    Block-Strassen decomposition for 4x4 x 4x8.
    
    View A (4x4) as 2x2 blocks of 2x2: [[A00, A01], [A10, A11]]
    View B (4x8) as 2x4 blocks of 2x2: [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    View C (4x8) as 2x4 blocks of 2x2: [[C00, C01, C02, C03], [C10, C11, C12, C13]]
    
    Apply Strassen's 7-term algorithm at the 2x2 block level.
    For each Strassen term, we have a 2x2 effective A-block and need to multiply
    against all 4 B-column-blocks (each 2x2).
    
    Total rank: 7 (Strassen terms) * 7 (rank per 2x2x2) * 4 (B column blocks) = 196?
    No, we can share: for each Strassen term k, the effective A is fixed.
    We need to compute (A_eff_k) * (B_col_j) for j=0..3.
    Each such product is 2x2 x 2x2, rank 7.
    Total: 7 * 7 * 4 = 196. That's worse than 98.
    
    Alternative: treat B as 4x4 twice. Rank 49*2 = 98.
    """
    # Use the two-4x4x4 approach: rank 98
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    # Build 4x4x4 Strassen decomposition
    A_eff_specs = [
        [(1, 0), (1, 3)], # P1: A00+A11
        [(1, 2), (1, 3)], # P2: A10+A11
        [(1, 0)],         # P3: A00
        [(1, 3)],         # P4: A11
        [(1, 0), (1, 1)], # P5: A00+A01
        [(1, 2), (-1, 0)],# P6: A10-A00
        [(1, 1), (-1, 3)] # P7: A01-A11
    ]
    
    B_eff_specs = [
        [(1, 0), (1, 3)], # P1: B00+B11
        [(1, 0)],         # P2: B00
        [(1, 1), (-1, 3)],# P3: B01-B11
        [(1, 2), (-1, 0)],# P4: B10-B00
        [(1, 3)],         # P5: B11
        [(1, 0), (1, 1)], # P6: B00+B01
        [(1, 2), (1, 3)]  # P7: B10+B11
    ]
    
    C_eff_specs = [
        [(1, 0), (1, 3)], # C00+C11
        [(1, 2), (-1, 3)],# C10-C11
        [(1, 1), (1, 3)], # C01+C11
        [(1, 0), (1, 2)], # C00+C10
        [(-1, 0), (1, 1)],# -C00+C01
        [(1, 3)],         # C11
        [(1, 0)]          # C00
    ]
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for half in range(2):
        base_col = half * 49
        for k in range(7):
            for m in range(7):
                col_idx = base_col + k * 7 + m
                
                # U contribution (A side)
                for coeff_a, a_blk in A_eff_specs[k]:
                    bi, bj = divmod(a_blk, 2)
                    for lr in range(2):
                        for lc in range(2):
                            gr = bi * 2 + lr
                            gc = bj * 2 + lc
                            g_idx = gr * 4 + gc
                            l_idx = lr * 2 + lc
                            U[g_idx, col_idx] += coeff_a * U_2[l_idx, m]
                            
                # V contribution (B side)
                for coeff_b, b_blk in B_eff_specs[k]:
                    bi, bj = divmod(b_blk, 2)
                    for lr in range(2):
                        for lc in range(2):
                            gr = bi * 2 + lr
                            gc = bj * 2 + lc
                            # Global B index: row * 8 + col
                            g_idx = gr * 8 + (gc + half * 4)
                            l_idx = lr * 2 + lc
                            V[g_idx, col_idx] += coeff_b * V_2[l_idx, m]
                            
                # W contribution (C side)
                for coeff_c, c_blk in C_eff_specs[k]:
                    bi, bj = divmod(c_blk, 2)
                    for lr in range(2):
                        for lc in range(2):
                            gr = bi * 2 + lr
                            gc = bj * 2 + lc
                            # W index: col * 4 + row
                            g_idx = (gc + half * 4) * 4 + gr
                            l_idx = lr * 2 + lc
                            W[g_idx, col_idx] += coeff_c * W_2[l_idx, m]
                            
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    U, V, W = _block_strassen_4x4x8()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
