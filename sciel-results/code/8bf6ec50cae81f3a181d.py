"""Strassen-based rank reduction for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_DEADLINE = time.time() + 140  # safety margin before 150s kill

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    
    Strategy:
    - Use Strassen's algorithm at the top level to decompose 4x4*4x4 into 7 terms
    - Each term is a 2x2*2x2 product, decomposed with schoolbook (8 rank-1 terms)
    - Total rank for 4x4*4x4: 7 * 8 = 56
    - For 4x4*4x8: duplicate the decomposition for the second 4 columns
    - Total rank: 56 * 2 = 112
    
    This is better than schoolbook (128) but not optimal. The known best is 98
    (using Strassen at both levels: 7*7=49 for 4x4*4x4, then *2 for 4x4*4x8).
    
    Let me implement the full Strassen decomposition with rank 98.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Build the rank-49 decomposition for 4x4*4x4 using nested Strassen.
    # Then duplicate for the second half of B columns to get rank 98 for 4x4*4x8.
    
    # For 4x4*4x4:
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3
    # C[i,k] for i,k in 0..3
    #
    # U[i*4+j, r] = coefficient of A[i,j] in term r
    # V[j*4+k, r] = coefficient of B[j,k] in term r
    # W[k*4+i, r] = coefficient of C[i,k] in term r
    
    # For 4x4*4x8:
    # U[i*4+j, r] for i,j in 0..3, r in 0..97
    # V[j*8+k, r] for j in 0..3, k in 0..7, r in 0..97
    # W[k*4+i, r] for k in 0..7, i in 0..3, r in 0..97
    
    # Step 1: Define the top-level Strassen terms for 4x4*4x4.
    # A blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    
    # The 7 top-level Strassen terms:
    # T0: X = A00+A11, Y = B00+B11
    # T1: X = A11, Y = B00+B11
    # T2: X = A00-A11, Y = B10+B11
    # T3: X = A00, Y = B00-B10
    # T4: X = A00+A01, Y = B11
    # T5: X = A10+A11, Y = B00-B01
    # T6: X = A01-A11, Y = B10+B11
    
    # Each term X*Y is a 2x2*2x2 product, decomposed with Strassen (7 sub-terms).
    # Total: 7 * 7 = 49 rank-1 terms for 4x4*4x4.
    
    # For each top-level term Tt, we need to:
    # 1. Express X_t and Y_t as 2x2 matrices of linear combinations of A and B entries.
    # 2. Apply Strassen 2x2*2x2 to X_t * Y_t to get 7 sub-terms.
    # 3. Each sub-term is a product of a linear combination of A entries and a linear
    #    combination of B entries, contributing to specific C entries with specific coefficients.
    
    # Let me represent the coefficients as dictionaries.
    # For a linear combination of A entries: {(i,j): coeff}
    # For a linear combination of B entries: {(j,k): coeff}
    # For contribution to C: {(i,k): coeff}
    
    # Top-level term definitions:
    # Each term is (X_coeffs, Y_coeffs, C_contrib) where:
    # X_coeffs: 2x2 matrix, each entry is a dict {(i,j): coeff} for A
    # Y_coeffs: 2x2 matrix, each entry is a dict {(j,k): coeff} for B
    # C_contrib: dict {block: coeff} where block is (br, bc) in 0,1 and coeff is the
    #            coefficient for that C block.
    
    # C block (br, bc) corresponds to C[2*br:2*br+2, 2*bc:2*bc+2]
    # C[0:2, 0:2] = C00, C[0:2, 2:4] = C01, C[2:4, 0:2] = C10, C[2:4, 2:4] = C11
    
    # Strassen C contributions:
    # C00 = M1 - M2 + M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M5 + M3 + M7
    
    # So:
    # T0 (M1): contributes to C00 with +1, C11 with +1
    # T1 (M2): contributes to C00 with -1, C10 with +1
    # T2 (M3): contributes to C01 with +1, C11 with +1
    # T3 (M4): contributes to C00 with +1, C10 with +1
    # T4 (M5): contributes to C01 with +1, C11 with -1
    # T5 (M6): contributes to C00 with +1
    # T6 (M7): contributes to C11 with +1
    
    # Now, for each term, define X and Y as 2x2 matrices.
    
    # Helper: create a 2x2 matrix of dicts for A blocks
    def A_block(block, sign=1):
        """Return a 2x2 matrix of dicts for A block with given sign."""
        if block == 0:  # A00
            return [
                {(0,0): sign, (0,1): sign},
                {(1,0): sign, (1,1): sign}
            ]
        elif block == 1:  # A01
            return [
                {(0,2): sign, (0,3): sign},
                {(1,2): sign, (1,3): sign}
            ]
        elif block == 2:  # A10
            return [
                {(2,0): sign, (2,1): sign},
                {(3,0): sign, (3,1): sign}
            ]
        elif block == 3:  # A11
            return [
                {(2,2): sign, (2,3): sign},
                {(3,2): sign, (3,3): sign}
            ]
    
    def B_block(block, sign=1):
        """Return a 2x2 matrix of dicts for B block with given sign."""
        if block == 0:  # B00
            return [
                {(0,0): sign, (0,1): sign},
                {(1,0): sign, (1,1): sign}
            ]
        elif block == 1:  # B01
            return [
                {(0,2): sign, (0,3): sign},
                {(1,2): sign, (1,3): sign}
            ]
        elif block == 2:  # B10
            return [
                {(2,0): sign, (2,1): sign},
                {(3,0): sign, (3,1): sign}
            ]
        elif block == 3:  # B11
            return [
                {(2,2): sign, (2,3): sign},
                {(3,2): sign, (3,3): sign}
            ]
    
    def add_matrices(M1, M2):
        """Add two 2x2 matrices of dicts."""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for key, val in M1[i][j].items():
                    result[i][j][key] = result[i][j].get(key, 0) + val
                for key, val in M2[i][j].items():
                    result[i][j][key] = result[i][j].get(key, 0) + val
        return result
    
    def scale_matrix(M, s):
        """Scale a 2x2 matrix of dicts by scalar s."""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for key, val in M[i][j].items():
                    result[i][j][key] = val * s
        return result
    
    # Define the 7 top-level terms.
    # Each term: (X, Y, C_contrib) where C_contrib is a dict {(br, bc): coeff}
    
    # T0: X = A00+A11, Y = B00+B11
    T0_X = add_matrices(A_block(0), A_block(3))
    T0_Y = add_matrices(B_block(0), B_block(3))
    T0_C = {(0,0): 1, (1,1): 1}
    
    # T1: X = A11, Y = B00+B11
    T1_X = A_block(3)
    T1_Y = add_matrices(B_block(0), B_block(3))
    T1_C = {(0,0): -1, (1,0): 1}
    
    # T2: X = A00-A11, Y = B10+B11
    T2_X = add_matrices(A_block(0), scale_matrix(A_block(3), -1))
    T2_Y = add_matrices(B_block(2), B_block(3))
    T2_C = {(0,1): 1, (1,1): 1}
    
    # T3: X = A00, Y = B00-B10
    T3_X = A_block(0)
    T3_Y = add_matrices(B_block(0), scale_matrix(B_block(2), -1))
    T3_C = {(0,0): 1, (1,0): 1}
    
    # T4: X = A00+A01, Y = B11
    T4_X = add_matrices(A_block(0), A_block(1))
    T4_Y = B_block(3)
    T4_C = {(0,1): 1, (1,1): -1}
    
    # T5: X = A10+A11, Y = B00-B01
    T5_X = add_matrices(A_block(2), A_block(3))
    T5_Y = add_matrices(B_block(0), scale_matrix(B_block(1), -1))
    T5_C = {(0,0): 1}
    
    # T6: X = A01-A11, Y = B10+B11
    T6_X = add_matrices(A_block(1), scale_matrix(A_block(3), -1))
    T6_Y = add_matrices(B_block(2), B_block(3))
    T6_C = {(1,1): 1}
    
    terms = [
        (T0_X, T0_Y, T0_C),
        (T1_X, T1_Y, T1_C),
        (T2_X, T2_Y, T2_C),
        (T3_X, T3_Y, T3_C),
        (T4_X, T4_Y, T4_C),
        (T5_X, T5_Y, T5_C),
        (T6_X, T6_Y, T6_C),
    ]
    
    # Now, for each top-level term, apply Strassen 2x2*2x2.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # where each xij and yij is a dict of coefficients.
    
    # Strassen sub-terms:
    # S1 = (x00+x11)(y00+y11)
    # S2 = x11(y00+y11)
    # S3 = (x00-x11)(y10+y11)
    # S4 = x00(y00-y10)
    # S5 = (x00+x01)y11
    # S6 = (x10+x11)(y00-y01)
    # S7 = (x01-x11)(y10+y11)
    
    # (X*Y)[0,0] = S1 - S2 + S4 + S6
    # (X*Y)[0,1] = S3 + S5
    # (X*Y)[1,0] = S2 + S4
    # (X*Y)[1,1] = S1 - S5 + S3 + S7
    
    # Each St is a product of two linear combinations, so it's a rank-1 term.
    # The contribution of (X*Y)[p,q] to C is determined by the C_contrib of the
    # top-level term.
    
    # Let me generate all 49 rank-1 terms.
    
    U_coeffs = []  # List of dicts {(i,j): coeff} for A
    V_coeffs = []  # List of dicts {(j,k): coeff} for B
    W_coeffs = []  # List of dicts {(i,k): coeff} for C
    
    for t, (X, Y, C_contrib) in enumerate(terms):
        x00, x01 = X[0][0], X[0][1]
        x10, x11 = X[1][0], X[1][1]
        y00, y01 = Y[0][0], Y[0][1]
        y10, y11 = Y[1][0], Y[1][1]
        
        # Compute the 7 Strassen sub-terms.
        # Each sub-term is (A_dict, B_dict) where A_dict is for X side, B_dict is for Y side.
        
        # S1 = (x00+x11)(y00+y11)
        S1_A = add_dicts(x00, x11)
        S1_B = add_dicts(y00, y11)
        
        # S2 = x11(y00+y11)
        S2_A = x11
        S2_B = add_dicts(y00, y11)
        
        # S3 = (x00-x11)(y10+y11)
        S3_A = add_dicts(x00, scale_dict(x11, -1))
        S3_B = add_dicts(y10, y11)
        
        # S4 = x00(y00-y10)
        S4_A = x00
        S4_B = add_dicts(y00, scale_dict(y10, -1))
        
        # S5 = (x00+x01)y11
        S5_A = add_dicts(x00, x01)
        S5_B = y11
        
        # S6 = (x10+x11)(y00-y01)
        S6_A = add_dicts(x10, x11)
        S6_B = add_dicts(y00, scale_dict(y01, -1))
        
        # S7 = (x01-x11)(y10+y11)
        S7_A = add_dicts(x01, scale_dict(x11, -1))
        S7_B = add_dicts(y10, y11)
        
        # Now, compute the 4 entries of X*Y and their contributions to C.
        # (X*Y)[0,0] = S1 - S2 + S4 + S6
        # (X*Y)[0,1] = S3 + S5
        # (X*Y)[1,0] = S2 + S4
        # (X*Y)[1,1] = S1 - S5 + S3 + S7
        
        # Each entry (p,q) of X*Y contributes to C block (br, bc) at position (p,q),
        # which maps to C[2*br+p, 2*bc+q].
        
        # For each of the 4 entries, and for each C block in C_contrib,
        # add the corresponding rank-1 terms.
        
        entries = [
            (0, 0, [S1, S2, S4, S6], [1, -1, 1, 1]),  # (X*Y)[0,0]
            (0, 1, [S3, S5], [1, 1]),                  # (X*Y)[0,1]
            (1, 0, [S2, S4], [1, 1]),                  # (X*Y)[1,0]
            (1, 1, [S1, S5, S3, S7], [1, -1, 1, 1]),   # (X*Y)[1,1]
        ]
        
        for p, q, sub_terms, coeffs in entries:
            for (br, bc), c_coeff in C_contrib.items():
                # This entry contributes to C[2*br+p, 2*bc+q] with coefficient c_coeff.
                for st, coeff in zip(sub_terms, coeffs):
                    A_dict, B_dict = st
                    total_coeff = c_coeff * coeff
                    
                    # Add this rank-1 term.
                    U_coeffs.append(A_dict)
                    V_coeffs.append(B_dict)
                    
                    W_dict = {}
                    ci = 2 * br + p
                    ck = 2 * bc + q
                    W_dict[(ci, ck)] = total_coeff
                    W_coeffs.append(W_dict)
    
    # Now, convert the coefficient dicts to the U, V, W matrices.
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49) for 4x4*4x4
    # W has shape (p*n, rank) = (16, 49) for 4x4*4x4
    
    rank_4x4 = len(U_coeffs)  # Should be 49
    
    U_4x4 = [[0] * rank_4x4 for _ in range(n * m)]
    V_4x4 = [[0] * rank_4x4 for _ in range(m * 4)]  # 4x4*4x4
    W_4x4 = [[0] * rank_4x4 for _ in range(4 * n)]
    
    for r in range(rank_4x4):
        for (i, j), coeff in U_coeffs[r].items():
            U_4x4[i * m + j][r] = coeff
        for (j, k), coeff in V_coeffs[r].items():
            V_4x4[j * 4 + k][r] = coeff
        for (i, k), coeff in W_coeffs[r].items():
            W_4x4[k * n + i][r] = coeff
    
    # Now, extend to 4x4*4x8.
    # B is 4x8, so split into B_left (4x4) and B_right (4x4).
    # C = [A*B_left, A*B_right]
    
    # U stays the same (16 x 49).
    # V becomes 32 x 98: first 49 columns use B_left, next 49 columns use B_right.
    # W becomes 32 x 98: first 49 columns contribute to C_left (4x4), next 49 to C_right (4x4).
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy U_4x4 to both halves.
    for i in range(n * m):
        for r in range(rank_4x4):
            U[i][r] = U_4x4[i][r]
            U[i][r + rank_4x4] = U_4x4[i][r]
    
    # For V: first half uses B_left (columns 0..3), second half uses B_right (columns 4..7).
    for j in range(4):
        for k in range(4):
            for r in range(rank_4x4):
                V[j * p + k][r] = V_4x4[j * 4 + k][r]
                V[j * p + (k + 4)][r + rank_4x4] = V_4x4[j * 4 + k][r]
    
    # For W: first half contributes to C_left (columns 0..3), second half to C_right (columns 4..7).
    for k in range(4):
        for i in range(n):
            for r in range(rank_4x4):
                W[k * n + i][r] = W_4x4[k * n + i][r]
                W[(k + 4) * n + i][r + rank_4x4] = W_4x4[k * n + i][r]
    
    return {"U": U, "V": V, "W": W}

def add_dicts(d1, d2):
    """Add two coefficient dictionaries."""
    result = {}
    for key, val in d1.items():
        result[key] = result.get(key, 0) + val
    for key, val in d2.items():
        result[key] = result.get(key, 0) + val
    return result

def scale_dict(d, s):
    """Scale a coefficient dictionary by scalar s."""
    return {key: val * s for key, val in d.items()}
# EVOLVE-BLOCK-END
