import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A_rows: list of 2 row indices in A (0-2)
    # A_cols: list of 2 col indices in A (0-3)
    # B_rows: list of 2 row indices in B (0-3)
    # B_cols: list of 2 col indices in B (0-3)
    # C_rows: list of 2 row indices in C (0-2)
    # C_cols: list of 2 col indices in C (0-3)
    def add_strassen(A_r, A_c, B_r, B_c, C_r, C_c):
        nonlocal col
        # Map local 0,1 to global indices
        a00 = A_r[0] * m + A_c[0]
        a01 = A_r[0] * m + A_c[1]
        a10 = A_r[1] * m + A_c[0]
        a11 = A_r[1] * m + A_c[1]
        
        b00 = B_r[0] * p + B_c[0]
        b01 = B_r[0] * p + B_c[1]
        b10 = B_r[1] * p + B_c[0]
        b11 = B_r[1] * p + B_c[1]
        
        c00 = C_c[0] * n + C_r[0]
        c01 = C_c[1] * n + C_r[0]
        c10 = C_c[0] * n + C_r[1]
        c11 = C_c[1] * n + C_r[1]
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        return col

    # Helper to add a 1x2x2 schoolbook block
    # A_row: 1 row index in A
    # A_cols: list of 2 col indices in A
    # B_rows: list of 2 row indices in B
    # B_cols: list of 2 col indices in B
    # C_row: 1 row index in C
    # C_cols: list of 2 col indices in C
    def add_schoolbook_1x2x2(A_r, A_c, B_r, B_c, C_r, C_c):
        nonlocal col
        a0 = A_r * m + A_c[0]
        a1 = A_r * m + A_c[1]
        
        b00 = B_r[0] * p + B_c[0]
        b01 = B_r[0] * p + B_c[1]
        b10 = B_r[1] * p + B_c[0]
        b11 = B_r[1] * p + B_c[1]
        
        c0 = C_c[0] * n + C_r
        c1 = C_c[1] * n + C_r
        
        # C0 = A0 B00 + A1 B10
        U[a0][col] += 1
        V[b00][col] += 1
        W[c0][col] += 1
        col += 1
        
        U[a1][col] += 1
        V[b10][col] += 1
        W[c0][col] += 1
        col += 1
        
        # C1 = A0 B01 + A1 B11
        U[a0][col] += 1
        V[b01][col] += 1
        W[c1][col] += 1
        col += 1
        
        U[a1][col] += 1
        V[b11][col] += 1
        W[c1][col] += 1
        col += 1
        
        return col

    # Block 1: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 0:2] x B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 3: A[2, 0:2] x B[0:2, 0:2] -> C[2, 0:2]
    add_schoolbook_1x2x2(2, [0, 1], [0, 1], [0, 1], 2, [0, 1])
    
    # Block 4: A[2, 0:2] x B[0:2, 2:4] -> C[2, 2:4]
    add_schoolbook_1x2x2(2, [0, 1], [0, 1], [2, 3], 2, [2, 3])
    
    # Block 5: A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 6: A[0:2, 2:4] x B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # Block 7: A[2, 2:4] x B[2:4, 0:2] -> C[2, 0:2]
    add_schoolbook_1x2x2(2, [2, 3], [2, 3], [0, 1], 2, [0, 1])
    
    # Block 8: A[2, 2:4] x B[2:4, 2:4] -> C[2, 2:4]
    add_schoolbook_1x2x2(2, [2, 3], [2, 3], [2, 3], 2, [2, 3])
    
    return {"U": U, "V": V, "W": W}
