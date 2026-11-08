"""Rewritten program with rank 56 decomposition using Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

DEADLINE = 150.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56: 8 independent 2x2*2x2 Strassen products
    
    # The multiplication C = A @ B where A is 2x4 and B is 4x8.
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    
    # We can split the sum over j into two sums:
    # C[i, k] = (sum_{j=0}^{1} A[i, j] * B[j, k]) + (sum_{j=2}^{3} A[i, j] * B[j, k])
    
    # Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # Let B1 = B[0:2, :] (2x8), B2 = B[2:4, :] (2x8)
    
    # C = A1 @ B1 + A2 @ B2
    
    # We can further split B into 4 blocks of 2x2:
    # B1 = [B11, B12, B13, B14] where B11 = B[0:2, 0:2], etc.
    # B2 = [B21, B22, B23, B24] where B21 = B[2:4, 0:2], etc.
    
    # C = A1 @ [B11, B12, B13, B14] + A2 @ [B21, B22, B23, B24]
    # C = [A1@B11 + A2@B21, A1@B12 + A2@B22, A1@B13 + A2@B23, A1@B14 + A2@B24]
    
    # Each term like A1@B11 is a 2x2 * 2x2 multiplication.
    # There are 8 such terms: 4 for the first part (A1@B1x) and 4 for the second (A2@B2x).
    
    # Wait, the above decomposition gives 8 independent 2x2*2x2 products.
    # Each 2x2*2x2 product has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    R = 56
    U = [[0] * R for _ in range(n * m)]  # 8 x 56
    V = [[0] * R for _ in range(m * p)]  # 32 x 56
    W = [[0] * R for _ in range(p * n)]  # 16 x 56
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2 * 2x2 product
    # A_block is 2x2, B_block is 2x2
    # A_block indices in U: (a_row, a_col) -> a_row*m + a_col
    # B_block indices in V: (b_row, b_col) -> b_row*p + b_col
    # C_block indices in W: (c_row, c_col) -> c_row*n + c_col
    
    def add_strassen(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        nonlocal col
        
        # A indices
        a00 = a_row_base * m + a_col_base
        a01 = a_row_base * m + a_col_base + 1
        a10 = (a_row_base + 1) * m + a_col_base
        a11 = (a_row_base + 1) * m + a_col_base + 1
        
        # B indices
        b00 = b_row_base * p + b_col_base
        b01 = b_row_base * p + b_col_base + 1
        b10 = (b_row_base + 1) * p + b_col_base
        b11 = (b_row_base + 1) * p + b_col_base + 1
        
        # C indices (W indices)
        c00 = c_row_base * n + c_col_base
        c01 = c_row_base * n + c_col_base + 1
        c10 = (c_row_base + 1) * n + c_col_base
        c11 = (c_row_base + 1) * n + c_col_base + 1
        
        # Strassen terms:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add 7 rank-1 terms.
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # U: a00=1, a11=1
        # V: b00=1, b11=1
        # W: c00=1, c01=0, c10=0, c11=-1 (wait, C00 gets +M1, C11 gets -M2+... let's be careful)
        
        # Let's expand the contributions to C00, C01, C10, C11 for each M.
        
        # M1 = (A00 + A11)(B00 + B11)
        # Contribution to C00: +1
        # Contribution to C01: 0
        # Contribution to C10: 0
        # Contribution to C11: -1
        
        # M2 = (A10 + A11)B00
        # Contribution to C00: 0
        # Contribution to C01: 0
        # Contribution to C10: +1
        # Contribution to C11: -1
        
        # M3 = A00(B10 - B11)
        # Contribution to C00: 0
        # Contribution to C01: +1
        # Contribution to C10: 0
        # Contribution to C11: +1
        
        # M4 = A11(B01 - B00)
        # Contribution to C00: +1
        # Contribution to C01: 0
        # Contribution to C10: 0
        # Contribution to C11: 0
        
        # M5 = (A00 + A01)B11
        # Contribution to C00: -1
        # Contribution to C01: +1
        # Contribution to C10: 0
        # Contribution to C11: 0
        
        # M6 = (A10 - A00)(B00 + B01)
        # Contribution to C00: 0
        # Contribution to C01: 0
        # Contribution to C10: 0
        # Contribution to C11: +1
        
        # M7 = (A01 - A11)(B10 + B11)
        # Contribution to C00: -1
        # Contribution to C01: 0
        # Contribution to C10: +1
        # Contribution to C11: 0
        
        # Let's verify:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += -1
        col += 1
        
        # Term 2: M2
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] += -1
        col += 1
        
        # Term 3: M3
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] += -1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 4: M4
        U[a11][col] += 1
        V[b01][col] += 1
        V[b00][col] += -1
        W[c00][col] += 1
        col += 1
        
        # Term 5: M5
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += -1
        W[c01][col] += 1
        col += 1
        
        # Term 6: M6
        U[a10][col] += 1
        U[a00][col] += -1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # Term 7: M7
        U[a01][col] += 1
        U[a11][col] += -1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += -1
        W[c10][col] += 1
        col += 1
        
    # Now, apply this to the 8 blocks.
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B13 = B[0:2, 4:6], B14 = B[0:2, 6:8]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4], B23 = B[2:4, 4:6], B24 = B[2:4, 6:8]
    
    # Block (A1, B11) -> C[:, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block (A2, B21) -> C[:, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block (A1, B12) -> C[:, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # Block (A2, B22) -> C[:, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block (A1, B13) -> C[:, 4:6]
    add_strassen(0, 0, 0, 4, 0, 4)
    
    # Block (A2, B23) -> C[:, 4:6]
    add_strassen(0, 2, 2, 4, 0, 4)
    
    # Block (A1, B14) -> C[:, 6:8]
    add_strassen(0, 0, 0, 6, 0, 6)
    
    # Block (A2, B24) -> C[:, 6:8]
    add_strassen(0, 2, 2, 6, 0, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
