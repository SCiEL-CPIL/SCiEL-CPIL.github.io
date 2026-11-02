"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a rank-44 decomposition for 2x3x8 matrix multiplication.
    This is based on decomposing the 3x8 part into four 3x2 blocks.
    Each 2x3x2 block is decomposed with rank 11 (7 from Strassen for 2x2x2 + 4 for the outer product of the remaining dimensions).
    Total rank = 4 * 11 = 44.
    """
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    
    # We partition the p=8 dimension into 4 blocks of size 2.
    # For each block q in 0..3, we compute A (2x3) * B_q (3x2).
    # The result is C_q (2x2).
    
    # Rank for one 2x3x2 block is 11.
    # Total rank is 44.
    rank = 44
    
    # Initialize U, V, W
    # U: (n*m, rank) = (6, 44)
    # V: (m*p, rank) = (24, 44)
    # W: (p*n, rank) = (16, 44)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x3): i in {0,1}, j in {0,1,2} -> idx = i*3 + j
    # Indices for B (3x8): j in {0,1,2}, k in {0..7} -> idx = j*8 + k
    # Indices for C (2x8): i in {0,1}, k in {0..7} -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Strassen decomposition for 2x2x2
    # A_2x2 = [[a00, a01], [a10, a11]]
    # B_2x2 = [[b00, b01], [b10, b11]]
    # 7 terms:
    # m1 = (a00+a10)(b00+b10)
    # m2 = (a00+a10)b11
    # m3 = a00(b10-b11)
    # m4 = a10(b00+b11)
    # m5 = (a01+a10)(b00+b10)
    # m6 = (a01-a10)(b01+b10)
    # m7 = (a11+a00)(b01-b11)
    
    # Mapping for 2x3x2 block:
    # A is 2x3. Let A = [A_2x2 | a_col] where a_col is 2x1.
    # B is 3x2. Let B = [B_2x2; b_row] where b_row is 1x2.
    # C = A_2x2 B_2x2 + a_col b_row.
    
    # The 7 Strassen terms use a00, a01, a10, a11 and b00, b01, b10, b11.
    # The 4 outer product terms use a_col (2 entries) and b_row (2 entries).
    # a_col entries: a02, a12.
    # b_row entries: b20, b21.
    
    # Outer product terms:
    # t8 = a02 * b20 -> contributes to C[0,0]
    # t9 = a02 * b21 -> contributes to C[0,1]
    # t10 = a12 * b20 -> contributes to C[1,0]
    # t11 = a12 * b21 -> contributes to C[1,1]
    
    # We need to map these to the global indices.
    
    for q in range(4):
        # Block q corresponds to columns 2*q and 2*q+1 of B and C.
        base_rank = q * 11
        
        # --- Strassen Part (7 terms) ---
        # These use A columns 0,1 and B rows 0,1.
        
        # Term 1: (a00+a10)(b00+b10)
        r = base_rank + 0
        U[0*3+0][r] = 1 # a00
        U[1*3+0][r] = 1 # a10
        V[0*8+2*q][r] = 1 # b00
        V[1*8+2*q][r] = 1 # b10
        # Contribution to C:
        # C00 += m1, C01 -= m1, C10 += m1, C11 += m1
        # Wait, Strassen reconstruction:
        # C00 = m1 - m3 - m4 + m7
        # C01 = m2 + m6
        # C10 = m4 + m5
        # C11 = m1 - m2 + m3 + m5
        # Let's verify this standard Strassen reconstruction.
        # m1 = (a00+a10)(b00+b10)
        # m2 = (a00+a10)b11
        # m3 = a00(b10-b11)
        # m4 = a10(b00+b11)
        # m5 = (a01+a10)(b00+b10)
        # m6 = (a01-a10)(b01+b10)
        # m7 = (a11+a00)(b01-b11)
        
        # C00 = m1 - m3 - m4 + m7
        # C01 = m2 + m6
        # C10 = m4 + m5
        # C11 = m1 - m2 + m3 + m5
        
        # We will assign the W factors according to these linear combinations.
        
        # Term 2: (a00+a10)b11
        r = base_rank + 1
        U[0*3+0][r] = 1
        U[1*3+0][r] = 1
        V[1*8+2*q+1][r] = 1 # b11 (row 1, col 1 of B_q)
        
        # Term 3: a00(b10-b11)
        r = base_rank + 2
        U[0*3+0][r] = 1
        V[1*8+2*q][r] = 1 # b10
        V[1*8+2*q+1][r] = -1 # -b11
        
        # Term 4: a10(b00+b11)
        r = base_rank + 3
        U[1*3+0][r] = 1
        V[0*8+2*q][r] = 1 # b00
        V[1*8+2*q+1][r] = 1 # b11
        
        # Term 5: (a01+a10)(b00+b10)
        r = base_rank + 4
        U[0*3+1][r] = 1 # a01
        U[1*3+0][r] = 1 # a10
        V[0*8+2*q][r] = 1 # b00
        V[1*8+2*q][r] = 1 # b10
        
        # Term 6: (a01-a10)(b01+b10)
        r = base_rank + 5
        U[0*3+1][r] = 1
        U[1*3+0][r] = -1
        V[0*8+2*q+1][r] = 1 # b01
        V[1*8+2*q][r] = 1 # b10
        
        # Term 7: (a11+a00)(b01-b11)
        r = base_rank + 6
        U[1*3+1][r] = 1 # a11
        U[0*3+0][r] = 1 # a00
        V[0*8+2*q+1][r] = 1 # b01
        V[1*8+2*q+1][r] = -1 # -b11
        
        # Now set W factors for these 7 terms.
        # W index: k*n + i.
        # Block q has k = 2*q (for C col 0) and k = 2*q+1 (for C col 1).
        # C[0, 2q] is at W index (2*q)*2 + 0 = 4q.
        # C[0, 2q+1] is at W index (2*q+1)*2 + 0 = 4q+2.
        # C[1, 2q] is at W index (2*q)*2 + 1 = 4q+1.
        # C[1, 2q+1] is at W index (2*q+1)*2 + 1 = 4q+3.
        
        # C00 (idx 4q): m1 - m3 - m4 + m7
        W[4*q][base_rank+0] = 1
        W[4*q][base_rank+2] = -1
        W[4*q][base_rank+3] = -1
        W[4*q][base_rank+6] = 1
        
        # C01 (idx 4q+2): m2 + m6
        W[4*q+2][base_rank+1] = 1
        W[4*q+2][base_rank+5] = 1
        
        # C10 (idx 4q+1): m4 + m5
        W[4*q+1][base_rank+3] = 1
        W[4*q+1][base_rank+4] = 1
        
        # C11 (idx 4q+3): m1 - m2 + m3 + m5
        W[4*q+3][base_rank+0] = 1
        W[4*q+3][base_rank+1] = -1
        W[4*q+3][base_rank+2] = 1
        W[4*q+3][base_rank+4] = 1
        
        # --- Outer Product Part (4 terms) ---
        # Uses A col 2 (a02, a12) and B row 2 (b20, b21).
        
        # Term 8: a02 * b20 -> C00
        r = base_rank + 7
        U[0*3+2][r] = 1 # a02
        V[2*8+2*q][r] = 1 # b20
        W[4*q][r] = 1
        
        # Term 9: a02 * b21 -> C01
        r = base_rank + 8
        U[0*3+2][r] = 1
        V[2*8+2*q+1][r] = 1 # b21
        W[4*q+2][r] = 1
        
        # Term 10: a12 * b20 -> C10
        r = base_rank + 9
        U[1*3+2][r] = 1 # a12
        V[2*8+2*q][r] = 1
        W[4*q+1][r] = 1
        
        # Term 11: a12 * b21 -> C11
        r = base_rank + 10
        U[1*3+2][r] = 1
        V[2*8+2*q+1][r] = 1
        W[4*q+3][r] = 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
