"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Strategy: Decompose 3x4x4 into 1x4x4 + 2x4x4
    # 1x4x4: rank 16 (schoolbook, optimal)
    # 2x4x4: Use Strassen on 2x2 blocks -> rank 28 (vs schoolbook 32)
    # Total rank: 44
    
    # For the 2x4x4 part:
    # Split A (2x4) into two 2x2 blocks: A[:,0:2], A[:,2:4]
    # Split B (4x4) into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # Result C (2x4) = A[:,0:2] * B[0:2,0:2] + A[:,0:2] * B[0:2,2:4] + A[:,2:4] * B[2:4,0:2] + A[:,2:4] * B[2:4,2:4]
    # = [A_left * B_top_left + A_right * B_bottom_left, A_left * B_top_right + A_right * B_bottom_right]
    
    # For each 2x2x2 multiplication, use Strassen (rank 7)
    
    # U has shape (n*m, rank) = (12, 44)
    # V has shape (m*p, rank) = (16, 44)
    # W has shape (p*n, rank) = (12, 44)
    
    total_rank = 16 + 4 * 7  # 16 for 1x4x4, 4 Strassen decompositions for 2x4x4
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Part 1: 1x4x4 schoolbook (rank 16)
    # A is 1x4, B is 4x4, C is 1x4
    # C[k] = sum_j A[j] * B[j,k]
    # In our convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # i=0 for the 1x4x4 part
    for j in range(4):
        for k in range(4):
            c = j * 4 + k
            U[0 * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + 0][c] = 1.0
    
    # Part 2: 2x4x4 using Strassen
    # A is 2x4: split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # B is 4x4: split into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2)
    # C is 2x4: C_L (2x2) = A_L * B_TL + A_R * B_BL, C_R (2x2) = A_L * B_TR + A_R * B_BR
    
    # Strassen for 2x2x2:
    # Given X (2x2) and Y (2x2), compute Z = X * Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # Strassen's 7 products:
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x11 - x10) * (y10 + y11)
    
    # Then:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # For each of the 4 2x2x2 multiplications in the 2x4x4 part:
    # 1. A_L * B_TL -> contributes to C_L
    # 2. A_R * B_BL -> contributes to C_L
    # 3. A_L * B_TR -> contributes to C_R
    # 4. A_R * B_BR -> contributes to C_R
    
    # Let's index the Strassen decompositions:
    # For multiplication #r (r=0,1,2,3):
    #   X is either A_L or A_R (rows 0-1 of A, cols 0-1 or 2-3)
    #   Y is either B_TL, B_BL, B_TR, or B_BR
    #   Z contributes to C_L or C_R
    
    strassen_configs = [
        # (x_row_offset, x_col_offset, y_row_offset, y_col_offset, z_row_offset, z_col_offset)
        # A_L * B_TL -> C_L: x=(0,0), y=(0,0), z=(0,0)
        (0, 0, 0, 0, 0, 0),
        # A_R * B_BL -> C_L: x=(0,2), y=(2,0), z=(0,0)
        (0, 2, 2, 0, 0, 0),
        # A_L * B_TR -> C_R: x=(0,0), y=(0,2), z=(0,2)
        (0, 0, 0, 2, 0, 2),
        # A_R * B_BR -> C_R: x=(0,2), y=(2,2), z=(0,2)
        (0, 2, 2, 2, 0, 2),
    ]
    
    for r, (x_ro, x_co, y_ro, y_co, z_ro, z_co) in enumerate(strassen_configs):
        base_c = 16 + r * 7  # start of this Strassen block in rank
        
        # Strassen M1: (x00 + x11) * (y00 + y11)
        # x00 = A[x_ro, x_co], x11 = A[x_ro+1, x_co+1]
        # y00 = B[y_ro, y_co], y11 = B[y_ro+1, y_co+1]
        # z00 += M1, z11 += M1
        
        # M1 factor
        U[x_ro * m + x_co][base_c + 0] = 1.0
        U[(x_ro + 1) * m + (x_co + 1)][base_c + 0] = 1.0
        V[y_ro * p + y_co][base_c + 0] = 1.0
        V[(y_ro + 1) * p + (y_co + 1)][base_c + 0] = 1.0
        W[z_co * n + z_ro][base_c + 0] = 1.0  # z00
        W[(z_co + 1) * n + (z_ro + 1)][base_c + 0] = 1.0  # z11
        
        # M2: (x10 + x11) * y00
        # x10 = A[x_ro+1, x_co], x11 = A[x_ro+1, x_co+1]
        # y00 = B[y_ro, y_co]
        # z10 += M2, z11 -= M2
        
        U[(x_ro + 1) * m + x_co][base_c + 1] = 1.0
        U[(x_ro + 1) * m + (x_co + 1)][base_c + 1] = 1.0
        V[y_ro * p + y_co][base_c + 1] = 1.0
        W[z_co * n + (z_ro + 1)][base_c + 1] = 1.0  # z10
        W[(z_co + 1) * n + (z_ro + 1)][base_c + 1] = -1.0  # z11
        
        # M3: x00 * (y01 - y11)
        # x00 = A[x_ro, x_co]
        # y01 = B[y_ro, y_co+1], y11 = B[y_ro+1, y_co+1]
        # z01 += M3, z11 += M3
        
        U[x_ro * m + x_co][base_c + 2] = 1.0
        V[y_ro * p + (y_co + 1)][base_c + 2] = 1.0
        V[(y_ro + 1) * p + (y_co + 1)][base_c + 2] = -1.0
        W[(z_co + 1) * n + z_ro][base_c + 2] = 1.0  # z01
        W[(z_co + 1) * n + (z_ro + 1)][base_c + 2] = 1.0  # z11
        
        # M4: x11 * (y10 - y00)
        # x11 = A[x_ro+1, x_co+1]
        # y10 = B[y_ro+1, y_co], y00 = B[y_ro, y_co]
        # z00 += M4, z10 += M4
        
        U[(x_ro + 1) * m + (x_co + 1)][base_c + 3] = 1.0
        V[(y_ro + 1) * p + y_co][base_c + 3] = 1.0
        V[y_ro * p + y_co][base_c + 3] = -1.0
        W[z_co * n + z_ro][base_c + 3] = 1.0  # z00
        W[z_co * n + (z_ro + 1)][base_c + 3] = 1.0  # z10
        
        # M5: (x00 + x01) * y11
        # x00 = A[x_ro, x_co], x01 = A[x_ro, x_co+1]
        # y11 = B[y_ro+1, y_co+1]
        # z00 -= M5, z01 += M5
        
        U[x_ro * m + x_co][base_c + 4] = 1.0
        U[x_ro * m + (x_co + 1)][base_c + 4] = 1.0
        V[(y_ro + 1) * p + (y_co + 1)][base_c + 4] = 1.0
        W[z_co * n + z_ro][base_c + 4] = -1.0  # z00
        W[(z_co + 1) * n + z_ro][base_c + 4] = 1.0  # z01
        
        # M6: (x10 - x00) * (y00 + y01)
        # x10 = A[x_ro+1, x_co], x00 = A[x_ro, x_co]
        # y00 = B[y_ro, y_co], y01 = B[y_ro, y_co+1]
        # z11 += M6
        
        U[(x_ro + 1) * m + x_co][base_c + 5] = 1.0
        U[x_ro * m + x_co][base_c + 5] = -1.0
        V[y_ro * p + y_co][base_c + 5] = 1.0
        V[y_ro * p + (y_co + 1)][base_c + 5] = 1.0
        W[(z_co + 1) * n + (z_ro + 1)][base_c + 5] = 1.0  # z11
        
        # M7: (x11 - x10) * (y10 + y11)
        # x11 = A[x_ro+1, x_co+1], x10 = A[x_ro+1, x_co]
        # y10 = B[y_ro+1, y_co], y11 = B[y_ro+1, y_co+1]
        # z00 += M7
        
        U[(x_ro + 1) * m + (x_co + 1)][base_c + 6] = 1.0
        U[(x_ro + 1) * m + x_co][base_c + 6] = -1.0
        V[(y_ro + 1) * p + y_co][base_c + 6] = 1.0
        V[(y_ro + 1) * p + (y_co + 1)][base_c + 6] = 1.0
        W[z_co * n + z_ro][base_c + 6] = 1.0  # z00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
