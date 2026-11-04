import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    # Convert to Fraction matrices
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i, j, k] == 1 for all i, j, k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = U_frac[i * m + j][c]
                    v_val = V_frac[j * p + k][c]
                    w_val = W_frac[k * n + i][c]
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x2_4x4():
    """
    Constructs a rank 18 decomposition for 3x4 * 4x4.
    
    Structure:
    A is 3x4. Split into A_top (2x4) and A_bot (1x4).
    B is 4x4. Split into B_left (4x2) and B_right (4x2).
    C is 3x4. Split into C_top_left (2x2), C_top_right (2x2), C_bot_left (1x2), C_bot_right (1x2).
    
    Top part: A_top * B_left -> C_top_left (2x2 * 2x2) -> Rank 7
    Top part: A_top * B_right -> C_top_right (2x2 * 2x2) -> Rank 7
    Bot part: A_bot * B_left -> C_bot_left (1x2 * 2x2) -> Rank 2 (dot products)
    Bot part: A_bot * B_right -> C_bot_right (1x2 * 2x2) -> Rank 2 (dot products)
    
    Total Rank = 7 + 7 + 2 + 2 = 18.
    """
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps for 2x2 * 2x2
    # Indices in local 2x2 grid: (0,0), (0,1), (1,0), (1,1)
    # U corresponds to A indices (i, j)
    # V corresponds to B indices (j, k)
    # W corresponds to C indices (i, k)
    
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11) -> C00 += 1, C11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11)B00 -> C10 += 1, C11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01 - B11) -> C01 += 1, C11 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10 - B00) -> C00 += 1, C10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01)B11 -> C01 += 1, C00 -= 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (A10 - A00)(B00 + B01) -> C11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11) -> C00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col = 0
    
    # Block 1: A_top (rows 0,1) * B_left (cols 0,1) -> C_top_left (rows 0,1, cols 0,1)
    # Global offsets:
    # A: i in [0,1], j in [0,1]
    # B: j in [0,1], k in [0,1]
    # C: i in [0,1], k in [0,1]
    
    a_offset_i, a_offset_j = 0, 0
    b_offset_j, b_offset_k = 0, 0
    c_offset_i, c_offset_k = 0, 0
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
        for (li, lj), val in u_coeffs.items():
            gi = li + a_offset_i
            gj = lj + a_offset_j
            U[gi * m + gj][col] += val
        
        for (lj, lk), val in v_coeffs.items():
            gj = lj + b_offset_j
            gk = lk + b_offset_k
            V[gj * p + gk][col] += val
        
        for (li, lk), val in w_coeffs.items():
            gi = li + c_offset_i
            gk = lk + c_offset_k
            W[gk * n + gi][col] += val
        
        col += 1
        
    # Block 2: A_top (rows 0,1) * B_right (cols 2,3) -> C_top_right (rows 0,1, cols 2,3)
    # Global offsets:
    # A: i in [0,1], j in [0,1]
    # B: j in [0,1], k in [2,3]
    # C: i in [0,1], k in [2,3]
    
    a_offset_i, a_offset_j = 0, 0
    b_offset_j, b_offset_k = 0, 2
    c_offset_i, c_offset_k = 0, 2
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
        for (li, lj), val in u_coeffs.items():
            gi = li + a_offset_i
            gj = lj + a_offset_j
            U[gi * m + gj][col] += val
        
        for (lj, lk), val in v_coeffs.items():
            gj = lj + b_offset_j
            gk = lk + b_offset_k
            V[gj * p + gk][col] += val
        
        for (li, lk), val in w_coeffs.items():
            gi = li + c_offset_i
            gk = lk + c_offset_k
            W[gk * n + gi][col] += val
        
        col += 1
        
    # Block 3: A_bot (row 2) * B_left (cols 0,1) -> C_bot_left (row 2, cols 0,1)
    # This is just dot products.
    # A[2, j] * B[j, k] -> C[2, k] for j in [0,1], k in [0,1]
    for j in [0, 1]:
        for k in [0, 1]:
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1

    # Block 4: A_bot (row 2) * B_right (cols 2,3) -> C_bot_right (row 2, cols 2,3)
    # This is just dot products.
    # A[2, j] * B[j, k] -> C[2, k] for j in [0,1], k in [2,3]
    for j in [0, 1]:
        for k in [2, 3]:
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 18 decomposition
    result_18 = _strassen_2x2_4x4()
    
    # Verify exactly
    ok = _verify_exact(result_18["U"], result_18["V"], result_18["W"], n, m, p)
    
    if ok:
        return result_18
    
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook()

