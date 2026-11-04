"""Matrix multiplication 3x4x6 - Block-Strassen decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 6]

def _add_term(U, V, W, col, u_idx, v_idx, w_idx, coeff=1.0):
    """Add a single term to the decomposition matrices."""
    U[u_idx, col] += coeff
    V[v_idx, col] += coeff
    W[w_idx, col] += coeff

def _decompose_1x2x3(U, V, W, start_col, a_i, a_js, b_js, b_ks, c_i, c_ks):
    """
    Decompose 1x2x3 multiplication.
    A is 1x2, B is 2x3, C is 1x3.
    a_i: global i index for A (and C)
    a_js: list of 2 global j indices for A
    b_js: list of 2 global j indices for B (must match a_js order? No, B is 2x3)
    Actually, B is defined by its rows (j) and cols (k).
    b_js: the 2 j indices for B's rows.
    b_ks: the 3 k indices for B's cols.
    c_i: global i index for C
    c_ks: list of 3 global k indices for C
    """
    col = start_col
    # A = [a0, a1]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = [c0, c1, c2]
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    # c2 = a0*b02 + a1*b12
    
    a_idx_0 = a_i * 4 + a_js[0]
    a_idx_1 = a_i * 4 + a_js[1]
    
    # Term 1: a0 * b00 * c0
    b_idx_00 = b_js[0] * 6 + b_ks[0]
    w_idx_0 = c_ks[0] * 3 + c_i
    _add_term(U, V, W, col, a_idx_0, b_idx_00, w_idx_0)
    col += 1
    
    # Term 2: a1 * b10 * c0
    b_idx_10 = b_js[1] * 6 + b_ks[0]
    _add_term(U, V, W, col, a_idx_1, b_idx_10, w_idx_0)
    col += 1
    
    # Term 3: a0 * b01 * c1
    b_idx_01 = b_js[0] * 6 + b_ks[1]
    w_idx_1 = c_ks[1] * 3 + c_i
    _add_term(U, V, W, col, a_idx_0, b_idx_01, w_idx_1)
    col += 1
    
    # Term 4: a1 * b11 * c1
    b_idx_11 = b_js[1] * 6 + b_ks[1]
    _add_term(U, V, W, col, a_idx_1, b_idx_11, w_idx_1)
    col += 1
    
    # Term 5: a0 * b02 * c2
    b_idx_02 = b_js[0] * 6 + b_ks[2]
    w_idx_2 = c_ks[2] * 3 + c_i
    _add_term(U, V, W, col, a_idx_0, b_idx_02, w_idx_2)
    col += 1
    
    # Term 6: a1 * b12 * c2
    b_idx_12 = b_js[1] * 6 + b_ks[2]
    _add_term(U, V, W, col, a_idx_1, b_idx_12, w_idx_2)
    col += 1
    
    return col

def _decompose_2x2x3(U, V, W, start_col, a_ii, a_js, b_js, b_ks, c_ii, c_ks):
    """
    Decompose 2x2x3 multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    a_ii: list of 2 global i indices for A
    a_js: list of 2 global j indices for A
    b_js: list of 2 global j indices for B
    b_ks: list of 3 global k indices for B
    c_ii: list of 2 global i indices for C
    c_ks: list of 3 global k indices for C
    """
    col = start_col
    # Split B into B1 (2x1) and B2 (2x2)
    # B1 uses b_ks[0]
    # B2 uses b_ks[1], b_ks[2]
    
    # Part 1: A * B1 (2x2x1)
    # C1 uses c_ks[0]
    # c10 = a00*b00 + a01*b10
    # c11 = a10*b00 + a11*b10
    
    a_idx_00 = a_ii[0] * 4 + a_js[0]
    a_idx_01 = a_ii[0] * 4 + a_js[1]
    a_idx_10 = a_ii[1] * 4 + a_js[0]
    a_idx_11 = a_ii[1] * 4 + a_js[1]
    
    b_idx_00 = b_js[0] * 6 + b_ks[0]
    b_idx_10 = b_js[1] * 6 + b_ks[0]
    
    w_idx_00 = c_ks[0] * 3 + c_ii[0]
    w_idx_10 = c_ks[0] * 3 + c_ii[1]
    
    # Term 1: a00 * b00 * c00
    _add_term(U, V, W, col, a_idx_00, b_idx_00, w_idx_00)
    col += 1
    # Term 2: a01 * b10 * c00
    _add_term(U, V, W, col, a_idx_01, b_idx_10, w_idx_00)
    col += 1
    # Term 3: a10 * b00 * c10
    _add_term(U, V, W, col, a_idx_10, b_idx_00, w_idx_10)
    col += 1
    # Term 4: a11 * b10 * c10
    _add_term(U, V, W, col, a_idx_11, b_idx_10, w_idx_10)
    col += 1
    
    # Part 2: A * B2 (2x2x2) - Strassen
    # B2 rows: b_js[0], b_js[1]
    # B2 cols: b_ks[1], b_ks[2]
    
    # Define indices for B2 and C2
    # B2 = [[b01, b02], [b11, b12]] (local)
    # b_idx_01 = b_js[0] * 6 + b_ks[1]
    # b_idx_02 = b_js[0] * 6 + b_ks[2]
    # b_idx_11 = b_js[1] * 6 + b_ks[1]
    # b_idx_12 = b_js[1] * 6 + b_ks[2]
    
    # C2 = [[c01, c02], [c11, c12]] (local)
    # w_idx_01 = c_ks[1] * 3 + c_ii[0]
    # w_idx_02 = c_ks[2] * 3 + c_ii[0]
    # w_idx_11 = c_ks[1] * 3 + c_ii[1]
    # w_idx_12 = c_ks[2] * 3 + c_ii[1]
    
    # Strassen M1: (a00+a11)(b01+b12) -> c01 += 1, c12 += 1
    # u: a00, a11
    # v: b01, b12
    # w: c01, c12
    _add_term(U, V, W, col, a_idx_00, b_idx_01, w_idx_01)
    _add_term(U, V, W, col, a_idx_11, b_idx_12, w_idx_12)
    # Note: v needs to be b01+b12? No, u and v are vectors.
    # u = [1, 0, 1, 0] (a00, a11)
    # v = [1, 0, 1, 0] (b01, b12) -> wait, b01 is (0,1), b12 is (1,2).
    # My _add_term adds to specific indices.
    # So for M1:
    # U[a00] += 1, U[a11] += 1
    # V[b01] += 1, V[b12] += 1
    # W[c01] += 1, W[c12] += 1
    # This is correct.
    col += 1
    
    # M2: (a10+a11)b01 -> c11 += 1
    _add_term(U, V, W, col, a_idx_10, b_idx_01, w_idx_11)
    _add_term(U, V, W, col, a_idx_11, b_idx_01, w_idx_11)
    col += 1
    
    # M3: a00(b02-b12) -> c02 += 1, c11 -= 1
    _add_term(U, V, W, col, a_idx_00, b_idx_02, w_idx_02)
    _add_term(U, V, W, col, a_idx_00, b_idx_12, w_idx_11, -1.0)
    col += 1
    
    # M4: a11(b10-b00) -> c11 += 1, c01 -= 1
    # Wait, Strassen M4 is a11(b10 - b00)?
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # My mapping:
    # a11 -> a_idx_00
    # a22 -> a_idx_11
    # a21 -> a_idx_10
    # a12 -> a_idx_01
    # b11 -> b_idx_01 (local 0,0 of B2)
    # b22 -> b_idx_12 (local 1,1 of B2)
    # b12 -> b_idx_02 (local 0,1 of B2)
    # b21 -> b_idx_11 (local 1,0 of B2)
    
    # M4: a22(b21-b11)
    _add_term(U, V, W, col, a_idx_11, b_idx_11, w_idx_11) # a22 * b21 * c21 (M4 adds to C21)
    _add_term(U, V, W, col, a_idx_11, b_idx_01, w_idx_01, -1.0) # a22 * b11 * c11 (M4 subtracts from C11)
    col += 1
    
    # M5: (a11+a12)b22 -> c02 += 1
    _add_term(U, V, W, col, a_idx_00, b_idx_12, w_idx_02)
    _add_term(U, V, W, col, a_idx_01, b_idx_12, w_idx_02)
    col += 1
    
    # M6: (a21-a11)(b11+b12) -> c01 += 1
    _add_term(U, V, W, col, a_idx_10, b_idx_01, w_idx_01)
    _add_term(U, V, W, col, a_idx_10, b_idx_02, w_idx_01)
    _add_term(U, V, W, col, a_idx_00, b_idx_01, w_idx_01, -1.0)
    _add_term(U, V, W, col, a_idx_00, b_idx_02, w_idx_01, -1.0)
    col += 1
    
    # M7: (a12-a22)(b21+b22) -> c12 += 1
    _add_term(U, V, W, col, a_idx_01, b_idx_11, w_idx_12)
    _add_term(U, V, W, col, a_idx_01, b_idx_12, w_idx_12)
    _add_term(U, V, W, col, a_idx_11, b_idx_11, w_idx_12, -1.0)
    _add_term(U, V, W, col, a_idx_11, b_idx_12, w_idx_12, -1.0)
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    rank = 68
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block TL: A[0,0:4] * B[0:4,0:3] -> C[0,0:3]
    # Split A into [0:2], [2:4]
    # Split B into [0:2], [2:4]
    # Sub 1: A[0,0:2] * B[0:2,0:3]
    col = _decompose_1x2x3(U, V, W, col, 
                            a_i=0, a_js=[0, 1], 
                            b_js=[0, 1], b_ks=[0, 1, 2], 
                            c_i=0, c_ks=[0, 1, 2])
    # Sub 2: A[0,2:4] * B[2:4,0:3]
    col = _decompose_1x2x3(U, V, W, col, 
                            a_i=0, a_js=[2, 3], 
                            b_js=[2, 3], b_ks=[0, 1, 2], 
                            c_i=0, c_ks=[0, 1, 2])
    
    # Block TR: A[0,0:4] * B[0:4,3:6] -> C[0,3:6]
    # Sub 1: A[0,0:2] * B[0:2,3:6]
    col = _decompose_1x2x3(U, V, W, col, 
                            a_i=0, a_js=[0, 1], 
                            b_js=[0, 1], b_ks=[3, 4, 5], 
                            c_i=0, c_ks=[3, 4, 5])
    # Sub 2: A[0,2:4] * B[2:4,3:6]
    col = _decompose_1x2x3(U, V, W, col, 
                            a_i=0, a_js=[2, 3], 
                            b_js=[2, 3], b_ks=[3, 4, 5], 
                            c_i=0, c_ks=[3, 4, 5])
    
    # Block BL: A[1:3,0:4] * B[0:4,0:3] -> C[1:3,0:3]
    # Sub 1: A[1:3,0:2] * B[0:2,0:3]
    col = _decompose_2x2x3(U, V, W, col, 
                            a_ii=[1, 2], a_js=[0, 1], 
                            b_js=[0, 1], b_ks=[0, 1, 2], 
                            c_ii=[1, 2], c_ks=[0, 1, 2])
    # Sub 2: A[1:3,2:4] * B[2:4,0:3]
    col = _decompose_2x2x3(U, V, W, col, 
                            a_ii=[1, 2], a_js=[2, 3], 
                            b_js=[2, 3], b_ks=[0, 1, 2], 
                            c_ii=[1, 2], c_ks=[0, 1, 2])
    
    # Block BR: A[1:3,0:4] * B[0:4,3:6] -> C[1:3,3:6]
    # Sub 1: A[1:3,0:2] * B[0:2,3:6]
    col = _decompose_2x2x3(U, V, W, col, 
                            a_ii=[1, 2], a_js=[0, 1], 
                            b_js=[0, 1], b_ks=[3, 4, 5], 
                            c_ii=[1, 2], c_ks=[3, 4, 5])
    # Sub 2: A[1:3,2:4] * B[2:4,3:6]
    col = _decompose_2x2x3(U, V, W, col, 
                            a_ii=[1, 2], a_js=[2, 3], 
                            b_js=[2, 3], b_ks=[3, 4, 5], 
                            c_ii=[1, 2], c_ks=[3, 4, 5])
    
    return {'U': U, 'V': V, 'W': W}
