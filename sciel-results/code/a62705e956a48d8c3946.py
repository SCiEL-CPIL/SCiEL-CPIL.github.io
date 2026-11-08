import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    total_rank = 49 + 16
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Strassen 2x2 Factors ---
    # U2, V2, W2 are 4x7 matrices
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Block definitions for Strassen
    # s: 0..6
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign) for each s? No, C is sum of s.
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # M0: (A00+A11)(B00+B11)
    # M1: (A10+A11)B00
    # M2: A00(B10-B11)
    # M3: A11(B01-B00)
    # M4: (A00+A10)B11
    # M5: (A10-A00)(B00+B01)
    # M6: (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    A_defs = [
        [(0, 1), (3, 1)],   # M0: A00, A11
        [(2, 1), (3, 1)],   # M1: A10, A11
        [(0, 1)],           # M2: A00
        [(3, 1)],           # M3: A11
        [(0, 1), (2, 1)],   # M4: A00, A10
        [(2, 1), (0, -1)],  # M5: A10, -A00
        [(1, 1), (3, -1)]   # M6: A01, -A11
    ]
    
    B_defs = [
        [(0, 1), (3, 1)],   # M0: B00, B11
        [(0, 1)],           # M1: B00
        [(2, 1), (3, -1)],  # M2: B10, -B11
        [(1, 1), (0, -1)],  # M3: B01, -B00
        [(3, 1)],           # M4: B11
        [(0, 1), (1, 1)],   # M5: B00, B01
        [(2, 1), (3, 1)]    # M6: B10, B11
    ]
    
    C_defs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00: M0, M3, -M4, M6
        [(2, 1), (4, 1)],                   # C01: M2, M4
        [(1, 1), (3, 1)],                   # C10: M1, M3
        [(0, 1), (1, -1), (2, 1), (5, 1)]   # C11: M0, -M1, M2, M5
    ]
    
    # Fill U, V, W for the 49 Strassen terms
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            
            # Fill U
            for (b_idx, sign) in A_defs[s]:
                # b_idx is 0..3 corresponding to block (0,0), (0,1), (1,0), (1,1)
                # U index: i_block * 2 + i_inner, j_block * 2 + j_inner
                # U2 index: i_inner * 2 + j_inner
                bi, bj = divmod(b_idx, 2)
                for i_inner in range(2):
                    for j_inner in range(2):
                        u_idx = (bi * 2 + i_inner) * 4 + (bj * 2 + j_inner)
                        u2_idx = i_inner * 2 + j_inner
                        U[u_idx][c] += sign * U2[u2_idx][t]
                        
            # Fill V
            for (b_idx, sign) in B_defs[s]:
                bi, bj = divmod(b_idx, 2)
                for i_inner in range(2):
                    for j_inner in range(2):
                        v_idx = (bi * 2 + i_inner) * 5 + (bj * 2 + j_inner)
                        v2_idx = i_inner * 2 + j_inner
                        V[v_idx][c] += sign * V2[v2_idx][t]
                        
            # Fill W
            for (c_idx, sign) in C_defs[c_idx // 7 if False else 0]: # Placeholder, need to iterate C_defs properly
                pass
                
            # Correct W filling:
            # W depends on which C block this term contributes to.
            # Actually, W is filled based on C_defs.
            # For each C block, if it includes M_s with sign, we add W2 vector to that block.
            for c_block_idx in range(4):
                # Check if M_s is in C_defs[c_block_idx]
                coeff = 0
                for (m_idx, m_sign) in C_defs[c_block_idx]:
                    if m_idx == s:
                        coeff = m_sign
                        break
                if coeff != 0:
                    ci, cj = divmod(c_block_idx, 2)
                    for i_inner in range(2):
                        for j_inner in range(2):
                            w_idx = (cj * 2 + j_inner) * 4 + (ci * 2 + i_inner) # k*n+i? No, W[k*n+i].
                            # W index: k_block * 2 + k_inner, i_block * 2 + i_inner
                            # W2 index: k_inner * 2 + i_inner
                            w2_idx = j_inner * 2 + i_inner
                            W[w_idx][c] += coeff * W2[w2_idx][t]

    # Part 2: 4x4x1 (Rank 16)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            c = 49 + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1 # k=4, i=i. Index 4*4+i = 16+i.
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
