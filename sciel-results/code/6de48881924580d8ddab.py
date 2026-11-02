import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for the rest.
    Target Rank: 68 (vs 72 schoolbook).
    """
    n, m, p = 3, 3, 8
    # Total rank estimate:
    # 4 blocks of 2x2x2 (Strassen rank 7 each) = 28
    # 2x1x8 (schoolbook) = 16
    # 1x2x8 (schoolbook) = 16
    # 1x1x8 (schoolbook) = 8
    # Total = 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A_11 * B_top part (4 blocks of 2x2x2 using Strassen)
    # A_11 is A[0:2, 0:2]
    # B_top is B[0:2, 0:8] split into 4 blocks of width 2
    
    for q in range(4):
        k_offset = 2 * q
        # Strassen 7 terms for this block
        # P1: (A00+A11)(B0e+B1h)
        # Indices: A rows 0,1 cols 0,1. B rows 0,1 cols k_offset, k_offset+1.
        
        # P1
        U[0*3+0][col] += 1; U[1*3+1][col] += 1
        V[0*8+k_offset][col] += 1; V[1*8+k_offset+1][col] += 1
        W[k_offset*3+0][col] += 1; W[(k_offset+1)*3+0][col] += 1
        W[k_offset*3+1][col] += 1; W[(k_offset+1)*3+1][col] += 1
        col += 1
        
        # P2: (A10+A11)B0e
        U[1*3+0][col] += 1; U[1*3+1][col] += 1
        V[0*8+k_offset][col] += 1
        W[k_offset*3+1][col] += 1
        col += 1
        
        # P3: A00(B0h-B1h)
        U[0*3+0][col] += 1
        V[0*8+k_offset+1][col] += 1; V[1*8+k_offset+1][col] -= 1
        W[(k_offset+1)*3+0][col] += 1
        col += 1
        
        # P4: A11(B1e-B0e)
        U[1*3+1][col] += 1
        V[1*8+k_offset][col] += 1; V[0*8+k_offset][col] -= 1
        W[k_offset*3+0][col] += 1; W[k_offset*3+1][col] += 1
        col += 1
        
        # P5: (A00+A01)B1h
        U[0*3+0][col] += 1; U[0*3+1][col] += 1
        V[1*8+k_offset+1][col] += 1
        W[(k_offset+1)*3+0][col] -= 1 # Wait, C00 = P1+P4-P5+P7. So W for P5 in C00 is -1.
        # But W is shared. We need to be careful.
        # The decomposition is sum_c U_c V_c W_c = T.
        # So for each c, we add to W the coefficient.
        # C00 gets -1 * P5. So W[k_offset*3+0][col] -= 1?
        # No, P5 contributes to C00 with coeff -1, C01 with coeff 1.
        # So W entries for this col c should be:
        # W[k_offset*3+0] -= 1
        # W[(k_offset+1)*3+0] += 1
        # Let's re-verify Strassen output formulas.
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # I already added W for P1, P2, P3, P4.
        # For P5:
        W[k_offset*3+0][col] -= 1
        W[(k_offset+1)*3+0][col] += 1
        col += 1
        
        # P6: (A10-A00)(B0e+B0h)
        U[1*3+0][col] += 1; U[0*3+0][col] -= 1
        V[0*8+k_offset][col] += 1; V[0*8+k_offset+1][col] += 1
        W[(k_offset+1)*3+1][col] += 1 # C11 gets +1 * P6
        col += 1
        
        # P7: (A01-A11)(B1e+B1h)
        U[0*3+1][col] += 1; U[1*3+1][col] -= 1
        V[1*8+k_offset][col] += 1; V[1*8+k_offset+1][col] += 1
        W[k_offset*3+0][col] += 1 # C00 gets +1 * P7
        col += 1
        
    # 2. A_12 * B_bot part (2x1x8 schoolbook)
    # A_12 is A[0:2, 2:3] (rows 0,1 col 2)
    # B_bot is B[2:3, 0:8] (row 2 cols 0..7)
    # C contribution to rows 0,1
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] += 1
            V[2*8+k][col] += 1
            W[k*3+i][col] += 1
            col += 1
            
    # 3. A_21 * B_top part (1x2x8 schoolbook)
    # A_21 is A[2:3, 0:2] (row 2 cols 0,1)
    # B_top is B[0:2, 0:8] (rows 0,1 cols 0..7)
    # C contribution to row 2
    for j in range(2):
        for k in range(8):
            U[2*3+j][col] += 1
            V[j*8+k][col] += 1
            W[k*3+2][col] += 1
            col += 1
            
    # 4. A_22 * B_bot part (1x1x8 schoolbook)
    # A_22 is A[2:3, 2:3] (row 2 col 2)
    # B_bot is B[2:3, 0:8] (row 2 cols 0..7)
    # C contribution to row 2
    for k in range(8):
        U[2*3+2][col] += 1
        V[2*8+k][col] += 1
        W[k*3+2][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
