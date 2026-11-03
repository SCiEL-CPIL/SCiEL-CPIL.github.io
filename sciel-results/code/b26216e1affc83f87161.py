"""Rank-66 decomposition for 3x4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # U: (n*m, rank) = (12, 66)
    # V: (m*p, rank) = (24, 66)
    # W: (p*n, rank) = (18, 66)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    RANK = 66
    U = [[0]*RANK for _ in range(n*m)]
    V = [[0]*RANK for _ in range(m*p)]
    W = [[0]*RANK for _ in range(p*n)]
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add a Strassen 2x2*2x2 block (rank 7)."""
        nonlocal col
        A00 = a_r*4 + a_c
        A01 = a_r*4 + a_c+1
        A10 = (a_r+1)*4 + a_c
        A11 = (a_r+1)*4 + a_c+1
        
        B00 = b_r*6 + b_c
        B01 = b_r*6 + b_c+1
        B10 = (b_r+1)*6 + b_c
        B11 = (b_r+1)*6 + b_c+1
        
        C00 = c_c*3 + c_r
        C01 = (c_c+1)*3 + c_r
        C10 = c_c*3 + (c_r+1)
        C11 = (c_c+1)*3 + (c_r+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1
        V[B11][col] -= 1
        W[C01][col] += 1
        W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1
        V[B00][col] -= 1
        W[C00][col] += 1
        W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1
        U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1
        W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1
        U[A00][col] -= 1
        V[B00][col] += 1
        V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1
        U[A11][col] -= 1
        V[B10][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_2x2_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add a 2x2*2x1 block (rank 4)."""
        nonlocal col
        A00 = a_r*4 + a_c
        A01 = a_r*4 + a_c+1
        A10 = (a_r+1)*4 + a_c
        A11 = (a_r+1)*4 + a_c+1
        B0 = b_r*6 + b_c
        B1 = (b_r+1)*6 + b_c
        C0 = c_c*3 + c_r
        C1 = c_c*3 + (c_r+1)
        
        U[A00][col] += 1
        V[B0][col] += 1
        W[C0][col] += 1
        col += 1
        U[A01][col] += 1
        V[B1][col] += 1
        W[C0][col] += 1
        col += 1
        U[A10][col] += 1
        V[B0][col] += 1
        W[C1][col] += 1
        col += 1
        U[A11][col] += 1
        V[B1][col] += 1
        W[C1][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add a 1x2*2x2 block (rank 4)."""
        nonlocal col
        A0 = a_r*4 + a_c
        A1 = a_r*4 + a_c+1
        B00 = b_r*6 + b_c
        B01 = b_r*6 + b_c+1
        B10 = (b_r+1)*6 + b_c
        B11 = (b_r+1)*6 + b_c+1
        C0 = c_c*3 + c_r
        C1 = (c_c+1)*3 + c_r
        
        U[A0][col] += 1
        V[B00][col] += 1
        W[C0][col] += 1
        col += 1
        U[A1][col] += 1
        V[B10][col] += 1
        W[C0][col] += 1
        col += 1
        U[A0][col] += 1
        V[B01][col] += 1
        W[C1][col] += 1
        col += 1
        U[A1][col] += 1
        V[B11][col] += 1
        W[C1][col] += 1
        col += 1
    
    def add_1x2_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add a 1x2*2x1 block (rank 2)."""
        nonlocal col
        A0 = a_r*4 + a_c
        A1 = a_r*4 + a_c+1
        B0 = b_r*6 + b_c
        B1 = (b_r+1)*6 + b_c
        C0 = c_c*3 + c_r
        
        U[A0][col] += 1
        V[B0][col] += 1
        W[C0][col] += 1
        col += 1
        U[A1][col] += 1
        V[B1][col] += 1
        W[C0][col] += 1
        col += 1
    
    # Partition:
    # A: 3x4 -> rows [0:2], [2:3]; cols [0:2], [2:4]
    # B: 4x6 -> rows [0:2], [2:4]; cols [0:2], [2:3], [3:5], [5:6]
    # C: 3x6 -> rows [0:2], [2:3]; cols [0:2], [2:3], [3:5], [5:6]
    
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    add_strassen(0, 0, 0, 0, 0, 0)  # 7
    add_strassen(0, 2, 2, 0, 0, 0)  # 7
    
    # C[0:2, 2:3] = A[0:2,0:2]*B[0:2,2:3] + A[0:2,2:4]*B[2:4,2:3]
    add_2x2_2x1(0, 0, 0, 2, 0, 2)  # 4
    add_2x2_2x1(0, 2, 2, 2, 0, 2)  # 4
    
    # C[0:2, 3:5] = A[0:2,0:2]*B[0:2,3:5] + A[0:2,2:4]*B[2:4,3:5]
    add_strassen(0, 0, 0, 3, 0, 3)  # 7
    add_strassen(0, 2, 2, 3, 0, 3)  # 7
    
    # C[0:2, 5:6] = A[0:2,0:2]*B[0:2,5:6] + A[0:2,2:4]*B[2:4,5:6]
    add_2x2_2x1(0, 0, 0, 5, 0, 5)  # 4
    add_2x2_2x1(0, 2, 2, 5, 0, 5)  # 4
    
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    add_1x2_2x2(2, 0, 0, 0, 2, 0)  # 4
    add_1x2_2x2(2, 2, 2, 0, 2, 0)  # 4
    
    # C[2:3, 2:3] = A[2:3,0:2]*B[0:2,2:3] + A[2:3,2:4]*B[2:4,2:3]
    add_1x2_2x1(2, 0, 0, 2, 2, 2)  # 2
    add_1x2_2x1(2, 2, 2, 2, 2, 2)  # 2
    
    # C[2:3, 3:5] = A[2:3,0:2]*B[0:2,3:5] + A[2:3,2:4]*B[2:4,3:5]
    add_1x2_2x2(2, 0, 0, 3, 2, 3)  # 4
    add_1x2_2x2(2, 2, 2, 3, 2, 3)  # 4
    
    # C[2:3, 5:6] = A[2:3,0:2]*B[0:2,5:6] + A[2:3,2:4]*B[2:4,5:6]
    add_1x2_2x1(2, 0, 0, 5, 2, 5)  # 2
    add_1x2_2x1(2, 2, 2, 5, 2, 5)  # 2
    
    # Total: 7+7+4+4+7+7+4+4+4+4+2+2+4+4+2+2 = 66
    assert col == 66, f"Expected 66 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
