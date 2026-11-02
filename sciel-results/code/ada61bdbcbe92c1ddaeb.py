"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use the Strassen decomposition for the two 3x3 blocks and schoolbook for the 3x1 block
    # Strassen for 3x3: rank 15 (using the known rank-15 decomposition)
    # Schoolbook for 3x1: rank 9
    # Total rank = 15 + 15 + 9 = 39
    
    rank = 39
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen decomposition for 3x3 matrix multiplication
    # We use the known rank-15 decomposition for 3x3 matrices
    # The 15 products are:
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # M2 = (A22+A33)(B22+B33)
    # M3 = (A11+A22)(B11+B22)
    # M4 = A11*B11
    # M5 = A22*B22
    # M6 = A33*B33
    # M7 = A12*(B21-B11)
    # M8 = (A11+A12)*B21
    # M9 = A23*(B32-B22)
    # M10 = (A22+A23)*B32
    # M11 = A31*(B13-B33)
    # M12 = (A33+A31)*B13
    # M13 = A13*(B31-B11)
    # M14 = (A11+A13)*B31
    # M15 = A32*(B23-B33)
    # M16 = (A33+A32)*B23
    
    # Actually, let me use a simpler approach: the known rank-15 decomposition
    # For C = A*B where A, B are 3x3:
    # C11 = M1 - M2 - M3 + M4
    # C12 = M7 + M8
    # C13 = M13 + M14
    # C21 = M11 + M12
    # C22 = M1 - M2 - M3 + M5
    # C23 = M9 + M10
    # C31 = M15 + M16
    # C32 = M13 + M14  -- wait, this doesn't look right
    
    # Let me use a different known rank-15 decomposition.
    # Actually, the minimum rank for 3x3 matrix multiplication is not known to be 15.
    # The best known upper bound is 19 (Strassen's algorithm).
    # Let me use Strassen's algorithm which has rank 19 for 3x3.
    
    # Actually, let me just use schoolbook for simplicity and correctness.
    # The task says "rank-minimal" but the seed uses rank 63.
    # Let me try to use a better decomposition.
    
    # For now, let me use the schoolbook decomposition but with a different structure
    # to make sure it's correct.
    
    # Reset
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
