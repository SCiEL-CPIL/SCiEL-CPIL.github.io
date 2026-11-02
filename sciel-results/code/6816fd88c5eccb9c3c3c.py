"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    - Use rank-49 Strassen decomposition for the 4x4 * 4x4 part
    - Use schoolbook for the 4x4 * 4x5 part
    - Total rank: 49 + 80 = 129
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 100
    
    # Build 4x4x4 Strassen rank-49 decomposition
    # A is 4x4, B_left is 4x4 (first 4 cols of B), C_left is 4x4 (first 4 cols of C)
    
    # Strassen level 1: 7 products of 2x2 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # Each Mt is a 2x2*2x2 product, decomposed with Strassen (7 scalar products)
    
    # A block indices (flat index = row*4+col):
    # A00: (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # A01: (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # A10: (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # A11: (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    A_blocks = {
        'A00': [0,1,4,5],
        'A01': [2,3,6,7],
        'A10': [8,9,12,13],
        'A11': [10,11,14,15],
    }
    
    # B block indices (flat index = row*4+col for the 4x4 part):
    B_blocks = {
        'B00': [0,1,4,5],
        'B01': [2,3,6,7],
        'B10': [8,9,12,13],
        'B11': [10,11,14,15],
    }
    
    # Level 1 Strassen products
    L1 = [
        ({'A00':1,'A11':1}, {'B00':1,'B11':1}),  # M1
        ({'A10':1,'A11':1}, {'B00':1}),           # M2
        ({'A00':1}, {'B01':1,'B11':-1}),          # M3
        ({'A11':1}, {'B10':1,'B00':-1}),          # M4
        ({'A00':1,'A01':1}, {'B11':1}),           # M5
        ({'A10':1,'A00':-1}, {'B00':1,'B01':1}),  # M6
        ({'A01':1,'A11':-1}, {'B10':1,'B11':1}),  # M7
    ]
    
    # Output combos: C_block -> {M_idx: coeff}
    C_combos = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},  # C00 = M1+M4-M5+M7
        (0,1): {2:1, 4:1},              # C01 = M3+M5
        (1,0): {1:1, 3:1},              # C10 = M2+M4
        (1,1): {0:1, 1:-1, 2:1, 5:1},  # C11 = M1-M2+M3+M6
    }
    
    # Strassen scalar products for 2x2*2x2:
    # Given P (2x2) and Q (2x2), the 7 products are:
    # S1 = (P00+P11)(Q00+Q11)
    # S2 = (P10+P11)Q00
    # S3 = P00(Q01-Q11)
    # S4 = P11(Q10-Q00)
    # S5 = (P00+P01)Q11
    # S6 = (P10-P00)(Q00+Q01)
    # S7 = (P01-P11)(Q10+Q11)
    #
    # Output:
    # R00 = S1+S4-S5+S7
    # R01 = S3+S5
    # R10 = S2+S4
    # R11 = S1-S2+S3+S6
    
    R_entry_coeffs = {
        (0,0): [1,0,0,1,-1,0,1],
        (0,1): [0,0,1,0,1,0,0],
        (1,0): [0,1,0,1,0,0,0],
        (1,1): [1,-1,1,0,0,1,0],
    }
    
    # Build all terms
    terms = []  # List of (U_dict, V_dict, W_dict)
    
    for t_idx, (A_combo, B_combo) in enumerate(L1):
        # Compute effective P and Q as dicts of index->coeff
        P = [[{}, {}], [{}, {}]]
        Q = [[{}, {}], [{}, {}]]
        
        for blk, coeff in A_combo.items():
            for pos, idx in enumerate(A_blocks[blk]):
                row, col = divmod(pos, 2)
                P[row][col][idx] = coeff
        
        for blk, coeff in B_combo.items():
            for pos, idx in enumerate(B_blocks[blk]):
                row, col = divmod(pos, 2)
                Q[row][col][idx] = coeff
        
        # Compute 7 Strassen scalar products
        def add_dicts(d1, d2, sign=1):
            r = dict(d1)
            for k, v in d2.items():
                r[k] = r.get(k, 0) + sign * v
            return r
        
        U_V = []
        # S1
        U_V.append((add_dicts(P[0][0], P[1][1]), add_dicts(Q[0][0], Q[1][1])))
        # S2
        U_V.append((add_dicts(P[1][0], P[1][1]), dict(Q[0][0])))
        # S3
        U_V.append((dict(P[0][0]), add_dicts(Q[0][1], Q[1][1], -1)))
        # S4
        U_V.append((dict(P[1][1]), add_dicts(Q[1][0], Q[0][0], -1)))
        # S5
        U_V.append((add_dicts(P[0][0], P[0][1]), dict(Q[1][1])))
        # S6
        U_V.append((add_dicts(P[1][0], P[0][0], -1), add_dicts(Q[0][0], Q[0][1])))
        # S7
        U_V.append((add_dicts(P[0][1], P[1][1], -1), add_dicts(Q[1][0], Q[1][1])))
        
        # For each output block that uses this Mt
        for c_pos, mt_coeffs in C_combos.items():
            if t_idx not in mt_coeffs:
                continue
            c_coeff = mt_coeffs[t_idx]
            c_row, c_col = c_pos
            
            for s_idx, (U_d, V_d) in enumerate(U_V):
                for di in range(2):
                    for dj in range(2):
                        rc = R_entry_coeffs[(di,dj)][s_idx]
                        if rc == 0:
                            continue
                        total_c = c_coeff * rc
                        w_idx = (2*c_col+dj)*4 + (2*c_row+di)
                        W_d = {w_idx: total_c}
                        terms.append((U_d, V_d, W_d))
    
    rank_44 = len(terms)
    
    # Build U, V, W for the 4x4x4 part
    U_44 = [[Fraction(0)] * rank_44 for _ in range(16)]
    V_44 = [[Fraction(0)] * rank_44 for _ in range(36)]
    W_44 = [[Fraction(0)] * rank_44 for _ in range(36)]
    
    for r, (U_d, V_d, W_d) in enumerate(terms):
        for idx, coeff in U_d.items():
            U_44[idx][r] += coeff
        for idx, coeff in V_d.items():
            if idx < 36:
                V_44[idx][r] += coeff
        for idx, coeff in W_d.items():
            if idx < 36:
                W_44[idx][r] += coeff
    
    if time.time() > deadline - 20:
        # Fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Schoolbook for 4x4 * 4x5 (columns 4-8 of B)
    rank_school = 80
    U_s = [[0] * rank_school for _ in range(16)]
    V_s = [[0] * rank_school for _ in range(36)]
    W_s = [[0] * rank_school for _ in range(36)]
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                r = (i * 4 + j) * 5 + (k - 4)
                U_s[i * 4 + j][r] = 1
                V_s[j * 9 + k][r] = 1
                W_s[k * 4 + i][r] = 1
    
    # Combine
    total_rank = rank_44 + rank_school
    U = [list(row) + [0] * rank_school for row in U_44]
    V = [list(row) + [0] * rank_school for row in V_44]
    W = [list(row) + [0] * rank_school for row in W_44]
    
    for r in range(rank_school):
        for i in range(16):
            U[i].append(U_s[i][r])
        for j in range(36):
            V[j].append(V_s[j][r])
        for k in range(36):
            W[k].append(W_s[k][r])
    
    # Convert Fraction to int/float
    def conv(x):
        if isinstance(x, Fraction):
            if x.denominator == 1:
                return x.numerator
            return float(x)
        return x
    
    U = [[conv(x) for x in row] for row in U]
    V = [[conv(x) for x in row] for row in V]
    W = [[conv(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
