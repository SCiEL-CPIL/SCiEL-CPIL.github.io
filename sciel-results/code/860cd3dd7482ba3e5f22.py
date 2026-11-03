import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses Strassen for 2x2 blocks (rank 7) and direct for 1x2*2x2 blocks (rank 4).
    Total rank = 3*2*4 + 3*2*7 = 24 + 42 = 66.
    """
    n, m, p = 3, 4, 6
    
    # Rank breakdown:
    # Row 0 of C (1x6): 3 column-pairs, each 1x2*2x2 rank 4 => 3*2*4 = 24
    # Rows 1-2 of C (2x6): 3 column-pairs, each 2x2*2x2 rank 7 => 3*2*7 = 42
    # Total = 66
    
    R = 66
    U = [[0] * R for _ in range(n * m)]  # (12, 66)
    V = [[0] * R for _ in range(m * p)]  # (24, 66)
    W = [[0] * R for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Strassen 2x2 * 2x2 multiplication (rank 7).
        a_r, a_c: top-left of A block
        b_r, b_c: top-left of B block
        c_r, c_c: top-left of C block (in W index: W[col*n + row])
        """
        nonlocal col
        # A indices in U (row-major: a_r*m + a_c)
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # B indices in V (row-major: b_r*p + b_c)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices in W (trace convention: W[k*n + i, r] where k=col, i=row)
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
    
    def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Direct 1x2 * 2x2 multiplication (rank 4).
        a_r, a_c: start of A row
        b_r, b_c: top-left of B block
        c_r, c_c: top-left of C row (in W index: W[col*n + row])
        """
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
        
        # C0 = A0*B00 + A1*B10
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # C1 = A0*B01 + A1*B11
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
    
    # Build the full decomposition
    # C has shape 3x6, partitioned as rows [1, 2] and cols [2, 2, 2]
    # A has shape 3x4, partitioned as rows [1, 2] and cols [2, 2]
    # B has shape 4x6, partitioned as rows [2, 2] and cols [2, 2, 2]
    
    # Row 0 of C (single row): 1x2 * 2x2 blocks
    for j in range(3):  # 3 column blocks of B
        for i in range(2):  # 2 column blocks of A
            add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
    
    # Rows 1-2 of C (2 rows): 2x2 * 2x2 Strassen blocks
    for j in range(3):  # 3 column blocks of B
        for i in range(2):  # 2 column blocks of A
            add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
    
    assert col == R, f"Expected rank {R}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
