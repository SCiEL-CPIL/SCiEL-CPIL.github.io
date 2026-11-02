"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # A is 3x3, B is 3x8
    # Split A into A00 (2x2), A01 (2x1), A10 (1x2), A11 (1x1)
    # Split B into 4 blocks of B0j (2x2) and B1j (1x2) for j=0..3
    
    # We will build U, V, W matrices
    # U shape: (n*m, rank) = (9, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (24, 44)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 by 2x2
    # M1 = (A00 + A11) * (B00 + B11) -> Not applicable directly as blocks are not square in the same way
    # Standard Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Here A00 is 2x2, B0j is 2x2.
    # Let A00 = [[a00, a01], [a10, a11]]
    # Let B0j = [[b00, b01], [b10, b11]]
    
    for j in range(4):
        # Strassen terms for A00 * B0j
        # Indices for A00:
        # a00: U[0], V[0], W[0] ? No, U, V, W are global.
        
        # We need to map local indices to global U, V, W.
        # A00 is rows 0,1 cols 0,1 of A.
        # U indices for A: i*m + j.
        # a00 -> U[0]
        # a01 -> U[1]
        # a10 -> U[2]
        # a11 -> U[3]
        
        # B0j is rows 0,1 cols 2j, 2j+1 of B.
        # V indices for B: j*p + k.
        # b00 -> V[2j]
        # b01 -> V[2j+1]
        # b10 -> V[p + 2j]
        # b11 -> V[p + 2j + 1]
        
        # W indices for C: k*n + i.
        # C0j is rows 0,1 cols 2j, 2j+1 of C.
        # c00 -> W[2j]
        # c01 -> W[2j+1]
        # c10 -> W[n + 2j]
        # c11 -> W[n + 2j + 1]
        
        # M1 = (A11 + A22)(B11 + B22)
        # In our local 2x2:
        # A11=a00, A22=a11, B11=b00, B22=b11
        # (a00+a11)(b00+b11)
        # U: a00, a11. V: b00, b11. W: c00, c11.
        # This requires 1 rank unit? No, Strassen uses 7 rank units for the whole 2x2 multiplication.
        # We must add 7 columns for this block.
        
        # M1
        U[0][col] = 1; V[2j][col] = 1; W[2j][col] = 1
        U[3][col] = 1; V[p + 2j + 1][col] = 1; W[2j][col] = 1
        U[0][col] = 1; V[p + 2j + 1][col] = 1; W[n + 2j + 1][col] = 1
        U[3][col] = 1; V[2j][col] = 1; W[n + 2j + 1][col] = 1
        col += 1
        
        # M2 = (A21 + A22)B11
        # (a10+a11)b00
        U[2][col] = 1; V[2j][col] = 1; W[n + 2j][col] = 1
        U[3][col] = 1; V[2j][col] = 1; W[n + 2j][col] = 1
        col += 1
        
        # M3 = A11(B12 - B22)
        # a00(b01-b11)
        U[0][col] = 1; V[2j+1][col] = 1; W[2j+1][col] = 1
        U[0][col] = 1; V[p + 2j + 1][col] = -1; W[2j+1][col] = -1
        col += 1
        
        # M4 = A22(B21 - B11)
        # a11(b10-b00)
        U[3][col] = 1; V[p + 2j][col] = 1; W[n + 2j + 1][col] = 1
        U[3][col] = 1; V[2j][col] = -1; W[n + 2j + 1][col] = -1
        col += 1
        
        # M5 = (A11 + A12)B22
        # (a00+a01)b11
        U[0][col] = 1; V[p + 2j + 1][col] = 1; W[n + 2j + 1][col] = 1
        U[1][col] = 1; V[p + 2j + 1][col] = 1; W[n + 2j + 1][col] = 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # (a10-a00)(b00+b01)
        U[2][col] = 1; V[2j][col] = 1; W[n + 2j][col] = 1
        U[0][col] = -1; V[2j][col] = -1; W[n + 2j][col] = -1
        U[2][col] = 1; V[2j+1][col] = 1; W[n + 2j][col] = 1
        U[0][col] = -1; V[2j+1][col] = -1; W[n + 2j][col] = -1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # (a01-a11)(b10+b11)
        U[1][col] = 1; V[p + 2j][col] = 1; W[2j+1][col] = 1
        U[3][col] = -1; V[p + 2j][col] = -1; W[2j+1][col] = -1
        U[1][col] = 1; V[p + 2j + 1][col] = 1; W[2j+1][col] = 1
        U[3][col] = -1; V[p + 2j + 1][col] = -1; W[2j+1][col] = -1
        col += 1
        
        # Nj = A01 * B1j
        # A01 is 2x1: [a02, a12]
        # B1j is 1x2: [b20, b21] (row 2 of B)
        # V indices: p + 2j, p + 2j + 1
        # W indices: 2j, 2j+1
        # Outer product
        U[4][col] = 1; V[p + 2j][col] = 1; W[2j][col] = 1
        U[4][col] = 1; V[p + 2j + 1][col] = 1; W[2j+1][col] = 1
        U[5][col] = 1; V[p + 2j][col] = 1; W[n + 2j][col] = 1
        U[5][col] = 1; V[p + 2j + 1][col] = 1; W[n + 2j + 1][col] = 1
        col += 1
        
        # Pj = A10 * B0j
        # A10 is 1x2: [a20, a21]
        # B0j is 2x2
        # Rank 2
        # C1j = [c20, c21]
        # c20 = a20*b00 + a21*b10
        # c21 = a20*b01 + a21*b11
        # Term 1: a20 * [b00, b01]
        U[6][col] = 1; V[2j][col] = 1; W[p*n + 2j][col] = 1
        U[6][col] = 1; V[2j+1][col] = 1; W[p*n + 2j + 1][col] = 1
        col += 1
        # Term 2: a21 * [b10, b11]
        U[7][col] = 1; V[p + 2j][col] = 1; W[p*n + 2j][col] = 1
        U[7][col] = 1; V[p + 2j + 1][col] = 1; W[p*n + 2j + 1][col] = 1
        col += 1
        
        # Qj = A11 * B1j
        # A11 is 1x1: a22
        # B1j is 1x2: [b20, b21]
        # Rank 1
        U[8][col] = 1; V[p + 2j][col] = 1; W[p*n + 2j][col] = 1
        U[8][col] = 1; V[p + 2j + 1][col] = 1; W[p*n + 2j + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
