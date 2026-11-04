import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 4 Strassen blocks (2x2x2x2) -> 4 * 7 = 28
    # 2 blocks (2x1x1x2) -> 2 * 1 = 2
    # 2 blocks (2x2x2x1) -> 2 * 1 = 2
    # 1 block (2x1x1x1) -> 1 * 1 = 1
    total_rank = 33
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients
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
    
    def add_strassen_block(j_start, k_start):
        nonlocal col
        # Local indices 0,1 map to global j_start+0, j_start+1 etc.
        
        # M1
        # A: A11, A22. Global A indices: i*m + (j_start+0), i*m + (j_start+1)
        # B: B11, B22. Global B indices: (j_start+0)*p + (k_start+0), (j_start+1)*p + (k_start+1)
        # C: C11, C22. Global C indices: (k_start+0)*n + 0, (k_start+0)*n + 1, (k_start+1)*n + 0, (k_start+1)*n + 1
        
        # A coeffs
        U[0*m + j_start][col] = 1
        U[1*m + j_start+1][col] = 1
        # B coeffs
        V[j_start*p + k_start][col] = 1
        V[(j_start+1)*p + k_start+1][col] = 1
        # C coeffs
        W[k_start*n + 0][col] = 1
        W[k_start*n + 1][col] = 1
        W[(k_start+1)*n + 0][col] = 1
        W[(k_start+1)*n + 1][col] = 1
        col += 1
        
        # M2
        # A: A21, A22. i=1. j_start, j_start+1
        # B: B11. j_start, k_start
        # C: C21, C22
        U[1*m + j_start][col] = 1
        U[1*m + j_start+1][col] = 1
        V[j_start*p + k_start][col] = 1
        W[(k_start+1)*n + 0][col] = 1 # C21 is i=1, k=k_start+1? No.
        # C indices:
        # C11: i=0, k=k_start
        # C12: i=0, k=k_start+1
        # C21: i=1, k=k_start
        # C22: i=1, k=k_start+1
        
        # Correction:
        # C21 is i=1, k=k_start. Index: k_start*n + 1.
        # C22 is i=1, k=k_start+1. Index: (k_start+1)*n + 1.
        
        W[k_start*n + 1][col] = 1
        W[(k_start+1)*n + 1][col] = 1
        col += 1
        
        # M3
        # A: A11. i=0, j_start
        # B: B12, B22. B12 is j_start, k_start+1. B22 is j_start+1, k_start+1.
        # C: C12, C22
        U[0*m + j_start][col] = 1
        V[j_start*p + k_start+1][col] = 1
        V[(j_start+1)*p + k_start+1][col] = -1
        W[k_start*n + 0][col] = 1 # C12? No, C12 is i=0, k=k_start+1.
        # C12 index: (k_start+1)*n + 0.
        # C22 index: (k_start+1)*n + 1.
        W[(k_start+1)*n + 0][col] = 1
        W[(k_start+1)*n + 1][col] = 1
        col += 1
        
        # M4
        # A: A22. i=1, j_start+1
        # B: B21, B11. B21 is j_start+1, k_start. B11 is j_start, k_start.
        # C: C11, C21
        U[1*m + j_start+1][col] = 1
        V[(j_start+1)*p + k_start][col] = 1
        V[j_start*p + k_start][col] = -1
        W[k_start*n + 0][col] = 1
        W[k_start*n + 1][col] = 1
        col += 1
        
        # M5
        # A: A11, A12. i=0. j_start, j_start+1
        # B: B22. j_start+1, k_start+1
        # C: C11, C12
        U[0*m + j_start][col] = 1
        U[0*m + j_start+1][col] = 1
        V[(j_start+1)*p + k_start+1][col] = 1
        W[k_start*n + 0][col] = -1 # C11
        W[(k_start+1)*n + 0][col] = 1 # C12
        col += 1
        
        # M6
        # A: A21, A11. i=1, i=0. j_start.
        # B: B11, B12. j_start. k_start, k_start+1
        # C: C22
        U[1*m + j_start][col] = 1
        U[0*m + j_start][col] = -1
        V[j_start*p + k_start][col] = 1
        V[j_start*p + k_start+1][col] = 1
        W[(k_start+1)*n + 1][col] = 1
        col += 1
        
        # M7
        # A: A12, A22. i=0, i=1. j_start+1
        # B: B21, B22. j_start+1. k_start, k_start+1
        # C: C11, C22
        U[0*m + j_start+1][col] = 1
        U[1*m + j_start+1][col] = -1
        V[(j_start+1)*p + k_start][col] = 1
        V[(j_start+1)*p + k_start+1][col] = 1
        W[k_start*n + 0][col] = 1
        W[(k_start+1)*n + 1][col] = 1
        col += 1

    # 4 Strassen blocks
    add_strassen_block(0, 0) # J1, K1
    add_strassen_block(2, 0) # J2, K1
    add_strassen_block(0, 2) # J1, K2
    add_strassen_block(2, 2) # J2, K2
    
    # Rectangular blocks
    
    # Block 5: J3, K1. A(2x1) B(1x2). j=4, k=0,1
    # Rank 1.
    # U: A11, A21. i=0,1. j=4.
    # V: B11, B12. j=4. k=0,1.
    # W: C11, C12, C21, C22.
    # C11: i=0, k=0.
    # C12: i=0, k=1.
    # C21: i=1, k=0.
    # C22: i=1, k=1.
    
    U[0*m + 4][col] = 1
    U[1*m + 4][col] = 1
    V[4*p + 0][col] = 1
    V[4*p + 1][col] = 1
    W[0*n + 0][col] = 1
    W[1*n + 0][col] = 1
    W[0*n + 1][col] = 1
    W[1*n + 1][col] = 1
    col += 1
    
    # Block 6: J3, K2. A(2x1) B(1x2). j=4, k=2,3
    U[0*m + 4][col] = 1
    U[1*m + 4][col] = 1
    V[4*p + 2][col] = 1
    V[4*p + 3][col] = 1
    W[2*n + 0][col] = 1
    W[3*n + 0][col] = 1
    W[2*n + 1][col] = 1
    W[3*n + 1][col] = 1
    col += 1
    
    # Block 7: J1, K3. A(2x2) B(2x1). j=0,1. k=4
    # Rank 1.
    # U: A11, A12, A21, A22.
    # V: B11, B21. j=0,1. k=4.
    # W: C11, C21. i=0,1. k=4.
    
    U[0*m + 0][col] = 1
    U[0*m + 1][col] = 1
    U[1*m + 0][col] = 1
    U[1*m + 1][col] = 1
    V[0*p + 4][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 0][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    # Block 8: J2, K3. A(2x2) B(2x1). j=2,3. k=4
    U[0*m + 2][col] = 1
    U[0*m + 3][col] = 1
    U[1*m + 2][col] = 1
    U[1*m + 3][col] = 1
    V[2*p + 4][col] = 1
    V[3*p + 4][col] = 1
    W[4*n + 0][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    # Block 9: J3, K3. A(2x1) B(1x1). j=4. k=4
    # Rank 1.
    U[0*m + 4][col] = 1
    U[1*m + 4][col] = 1
    V[4*p + 4][col] = 1
    W[4*n + 0][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
