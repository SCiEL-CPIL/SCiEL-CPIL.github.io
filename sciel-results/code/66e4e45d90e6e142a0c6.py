import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    # Target rank is 66 based on the partition:
    # A (3x4) -> rows 1+2, cols 2+2
    # B (4x6) -> rows 2+2, cols 2+2+2
    # C (3x6)
    # Row 0 of C: 3 pairs of (1x2 * 2x2) blocks. Each 1x2*2x2 is rank 4. Total 3*2*4 = 24.
    # Rows 1-2 of C: 3 pairs of (2x2 * 2x2) blocks. Each 2x2*2x2 is rank 7 (Strassen). Total 3*2*7 = 42.
    # Total rank = 24 + 42 = 66.
    
    R = 66
    U = [[0] * R for _ in range(n * m)]  # (12, 66)
    V = [[0] * R for _ in range(m * p)]  # (24, 66)
    W = [[0] * R for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block at given offsets."""
        nonlocal col
        # A indices
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # B indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (Trace convention: W[k*n + i, r])
        # c_r is row i, c_c is col k
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        # A indices (1x2)
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        # B indices (2x2)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (1x2)
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
    
    # Build decomposition:
    # C[0,0:2] = A[0,0:2]*B[0:2,0:2] + A[0,2:4]*B[2:4,0:2]
    add_1x2_2x2(0, 0, 0, 0, 0, 0)   # 1x2*2x2, rank 4
    add_1x2_2x2(0, 2, 2, 0, 0, 0)   # 1x2*2x2, rank 4
    
    # C[0,2:4] = A[0,0:2]*B[0:2,2:4] + A[0,2:4]*B[2:4,2:4]
    add_1x2_2x2(0, 0, 0, 2, 0, 2)   # 1x2*2x2, rank 4
    add_1x2_2x2(0, 2, 2, 2, 0, 2)   # 1x2*2x2, rank 4
    
    # C[0,4:6] = A[0,0:2]*B[0:2,4:6] + A[0,2:4]*B[2:4,4:6]
    add_1x2_2x2(0, 0, 0, 4, 0, 4)   # 1x2*2x2, rank 4
    add_1x2_2x2(0, 2, 2, 4, 0, 4)   # 1x2*2x2, rank 4
    
    # C[1:3,0:2] = A[1:3,0:2]*B[0:2,0:2] + A[1:3,2:4]*B[2:4,0:2]
    add_strassen(1, 0, 0, 0, 1, 0)   # 2x2*2x2, rank 7
    add_strassen(1, 2, 2, 0, 1, 0)   # 2x2*2x2, rank 7
    
    # C[1:3,2:4] = A[1:3,0:2]*B[0:2,2:4] + A[1:3,2:4]*B[2:4,2:4]
    add_strassen(1, 0, 0, 2, 1, 2)   # 2x2*2x2, rank 7
    add_strassen(1, 2, 2, 2, 1, 2)   # 2x2*2x2, rank 7
    
    # C[1:3,4:6] = A[1:3,0:2]*B[0:2,4:6] + A[1:3,2:4]*B[2:4,4:6]
    add_strassen(1, 0, 0, 4, 1, 4)   # 2x2*2x2, rank 7
    add_strassen(1, 2, 2, 4, 1, 4)   # 2x2*2x2, rank 7
    
    assert col == 66, f"Expected 66 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
