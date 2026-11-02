"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # We use a block decomposition to reduce the rank from 64 (schoolbook) to 56 (Strassen-based).
    # A (2x4) = [AL | AR] where AL, AR are 2x2
    # B (4x8) = [BT; BB] where BT, BB are 2x8
    # C = AL*BT + AR*BB
    # BT = [B00 B01 B02 B03] where Bij are 2x2
    # BB = [B10 B11 B12 B13] where Bij are 2x2
    # C = [AL*B00 + AR*B10, AL*B01 + AR*B11, AL*B02 + AR*B12, AL*B03 + AR*B13]
    # Each 2x2 multiplication can be done in rank 7 using Strassen.
    # Total rank = 8 * 7 = 56.
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 multiplication
    # A = [a b; c d], B = [e f; g h]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 multiplication A * B = C:
    # U has 4 rows (a,b,c,d), V has 4 rows (e,f,g,h), W has 4 rows (C00,C01,C10,C11)
    # We need to express M1..M7 as bilinear forms.
    
    strassen_U = []
    strassen_V = []
    strassen_W = []
    
    # M1 = (a+d)(e+h)
    strassen_U.extend([[1, 0, 0, 1]])  # a+d
    strassen_V.extend([[1, 0, 0, 1]])  # e+h
    strassen_W.extend([[1, 0, 0, 0]])  # C00 += 1*M1
    
    # M2 = (c+d)e
    strassen_U.extend([[0, 0, 1, 1]])  # c+d
    strassen_V.extend([[1, 0, 0, 0]])  # e
    strassen_W.extend([[0, 0, 1, 0]])  # C10 += 1*M2
    
    # M3 = a(f-h)
    strassen_U.extend([[1, 0, 0, 0]])  # a
    strassen_V.extend([[0, 1, 0, -1]]) # f-h
    strassen_W.extend([[0, 1, 0, 1]])  # C01 += 1*M3, C11 += 1*M3
    
    # M4 = d(g-e)
    strassen_U.extend([[0, 0, 0, 1]])  # d
    strassen_V.extend([[0, 0, 1, -1]]) # g-e
    strassen_W.extend([[-1, 0, 1, 0]]) # C00 += -1*M4, C10 += 1*M4
    
    # M5 = (a+b)h
    strassen_U.extend([[1, 1, 0, 0]])  # a+b
    strassen_V.extend([[0, 0, 0, 1]])  # h
    strassen_W.extend([[1, 1, 0, 0]])  # C00 += 1*M5, C01 += 1*M5
    
    # M6 = (c-a)(e+g)
    strassen_U.extend([[-1, 0, 1, 0]]) # c-a
    strassen_V.extend([[1, 0, 1, 0]])  # e+g
    strassen_W.extend([[0, 0, 0, 1]])  # C11 += 1*M6
    
    # M7 = (b-d)(f+h)
    strassen_U.extend([[0, 1, 0, -1]]) # b-d
    strassen_V.extend([[0, 1, 0, 1]])  # f+h
    strassen_W.extend([[-1, 0, 0, 0]]) # C00 += -1*M7
    
    # Now we need to assemble the full 2x4 * 4x8 multiplication.
    # We have 8 independent 2x2 multiplications.
    # Let's index them as (block, sub-block) where block in {L, R} and sub-block in {0, 1, 2, 3}
    # But actually, the structure is:
    # C_col_j = AL*B0j + AR*B1j
    # This is a sum of two 2x2 products.
    # We can treat each AL*B0j and AR*B1j as separate rank-7 components, and add their W contributions.
    
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # A is 2x4, indexed as A[i, j] where i in 0..1, j in 0..3
    # U rows correspond to A[i*m + j] = A[i, j]
    # AL = A[:, 0:2], AR = A[:, 2:4]
    
    # B is 4x8, indexed as B[i, j] where i in 0..3, j in 0..7
    # V rows correspond to B[i*p + j] = B[i, j]
    # BT = B[0:2, :], BB = B[2:4, :]
    # B0j = B[0:2, j*2:j*2+2], B1j = B[2:4, j*2:j*2+2]
    
    # C is 2x8, indexed as C[i, j] where i in 0..1, j in 0..7
    # W rows correspond to C[k*n + i] where k is column index?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i, where k is column (0..7) and i is row (0..1).
    # So W[k*n + i] corresponds to C[i, k].
    
    # For each of the 8 sub-multiplications:
    # We have (A_sub, B_sub) -> C_sub
    # A_sub is 2x2, B_sub is 2x2, C_sub is 2x2
    
    # Let's define the 8 sub-multiplications explicitly.
    # 1. AL * B00 -> C_part0 (contributes to C[:, 0:2])
    # 2. AL * B01 -> C_part1 (contributes to C[:, 2:4])
    # 3. AL * B02 -> C_part2 (contributes to C[:, 4:6])
    # 4. AL * B03 -> C_part3 (contributes to C[:, 6:8])
    # 5. AR * B10 -> C_part4 (contributes to C[:, 0:2])
    # 6. AR * B11 -> C_part5 (contributes to C[:, 2:4])
    # 7. AR * B12 -> C_part6 (contributes to C[:, 4:6])
    # 8. AR * B13 -> C_part7 (contributes to C[:, 6:8])
    
    # For each sub-multiplication, we use the Strassen decomposition.
    # We need to map the local indices (a,b,c,d) and (e,f,g,h) to global indices.
    
    sub_mults = []
    # (A_rows, A_cols, B_rows, B_cols, C_cols)
    # A_rows: which rows of A (0..1)
    # A_cols: which cols of A (0..3)
    # B_rows: which rows of B (0..3)
    # B_cols: which cols of B (0..7)
    # C_cols: which cols of C (0..7)
    
    # AL is A[:, 0:2], AR is A[:, 2:4]
    # B0j is B[0:2, j*2:j*2+2], B1j is B[2:4, j*2:j*2+2]
    
    for j in range(4):
        # AL * B0j
        sub_mults.append((
            [0, 1],      # A rows
            [0, 1],      # A cols (local to AL) -> global cols 0,1
            [0, 1],      # B rows
            [j*2, j*2+1],# B cols
            [j*2, j*2+1] # C cols
        ))
        # AR * B1j
        sub_mults.append((
            [0, 1],      # A rows
            [2, 3],      # A cols (global)
            [2, 3],      # B rows
            [j*2, j*2+1],# B cols
            [j*2, j*2+1] # C cols
        ))
    
    col_offset = 0
    for sm in sub_mults:
        A_rows, A_cols, B_rows, B_cols, C_cols = sm
        
        # Map local indices (0,1,2,3) to global indices
        # Local (0,0) -> global (A_rows[0], A_cols[0])
        # Local (0,1) -> global (A_rows[0], A_cols[1])
        # Local (1,0) -> global (A_rows[1], A_cols[0])
        # Local (1,1) -> global (A_rows[1], A_cols[1])
        
        # Similarly for B and C.
        
        for k in range(7):
            # Strassen_U[k] is [u0, u1, u2, u3] corresponding to (a, b, c, d)
            # a = A[0,0] local, b = A[0,1] local, c = A[1,0] local, d = A[1,1] local
            # Global indices for U:
            # a -> U[A_rows[0] * m + A_cols[0]]
            # b -> U[A_rows[0] * m + A_cols[1]]
            # c -> U[A_rows[1] * m + A_cols[0]]
            # d -> U[A_rows[1] * m + A_cols[1]]
            
            u_vals = strassen_U[k]
            v_vals = strassen_V[k]
            w_vals = strassen_W[k]
            
            # U contributions
            U[A_rows[0] * m + A_cols[0]][col_offset] += u_vals[0]
            U[A_rows[0] * m + A_cols[1]][col_offset] += u_vals[1]
            U[A_rows[1] * m + A_cols[0]][col_offset] += u_vals[2]
            U[A_rows[1] * m + A_cols[1]][col_offset] += u_vals[3]
            
            # V contributions
            # e = B[0,0] local, f = B[0,1] local, g = B[1,0] local, h = B[1,1] local
            # Global indices for V:
            # e -> V[B_rows[0] * p + B_cols[0]]
            # f -> V[B_rows[0] * p + B_cols[1]]
            # g -> V[B_rows[1] * p + B_cols[0]]
            # h -> V[B_rows[1] * p + B_cols[1]]
            
            V[B_rows[0] * p + B_cols[0]][col_offset] += v_vals[0]
            V[B_rows[0] * p + B_cols[1]][col_offset] += v_vals[1]
            V[B_rows[1] * p + B_cols[0]][col_offset] += v_vals[2]
            V[B_rows[1] * p + B_cols[1]][col_offset] += v_vals[3]
            
            # W contributions
            # C00 = C[0,0] local, C01 = C[0,1] local, C10 = C[1,0] local, C11 = C[1,1] local
            # Global indices for W:
            # C00 -> W[C_cols[0] * n + 0]
            # C01 -> W[C_cols[1] * n + 0]
            # C10 -> W[C_cols[0] * n + 1]
            # C11 -> W[C_cols[1] * n + 1]
            
            W[C_cols[0] * n + 0][col_offset] += w_vals[0]
            W[C_cols[1] * n + 0][col_offset] += w_vals[1]
            W[C_cols[0] * n + 1][col_offset] += w_vals[2]
            W[C_cols[1] * n + 1][col_offset] += w_vals[3]
            
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
