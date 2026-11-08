import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 9 x 33
    V = [[0] * rank for _ in range(m * p)]  # 12 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Part 1: 3x3 x 3x3 (first 3 columns of B)
    # B indices: 0,1,2 (col 0), 3,4,5 (col 1), 6,7,8 (col 2)
    # A indices: 0,1,2 (row 0), 3,4,5 (row 1), 6,7,8 (row 2)
    # C indices: 0,1,2 (col 0), 3,4,5 (col 1), 6,7,8 (col 2)
    
    # Strassen for A11 (0,1,3,4) * B11 (0,1,4,5) -> C11 (0,1,3,4)
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    # M4
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # A12 (2,5) * B21 (6,7) -> C12 (3,4) part? No, C11 part?
    # A12 is A[0,2], A[1,2]. B21 is B[2,0], B[2,1].
    # Result is C[0,0], C[0,1], C[1,0], C[1,1].
    # Indices: A: 2,5. B: 6,7. C: 0,1,3,4.
    # T1: A[0,2]B[2,0] -> C[0,0]
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    # T2: A[0,2]B[2,1] -> C[0,1]
    U[2][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    # T3: A[1,2]B[2,0] -> C[1,0]
    U[5][col] = 1
    V[6][col] = 1
    W[1][col] = 1
    col += 1
    # T4: A[1,2]B[2,1] -> C[1,1]
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # A11 (0,1,3,4) * B12 (2,3,6,7) -> C12 (3,4) part?
    # B12 is B[0,2], B[1,2], B[0,1]? No.
    # B12 is cols 2 of B11? No.
    # In 3x3 split, B12 is B[0,2], B[1,2]. (Col 2, Rows 0,1)
    # Indices: B: 2,3.
    # A11: 0,1,3,4.
    # Result C[0,2], C[1,2]. Indices C: 6,7.
    # Wait, C12 is C[0,2], C[1,2].
    # T1: A[0,0]B[0,2] -> C[0,2]
    U[0][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    # T2: A[0,1]B[1,2] -> C[0,2]
    U[1][col] = 1
    V[3][col] = 1
    W[6][col] = 1
    col += 1
    # T3: A[1,0]B[0,2] -> C[1,2]
    U[3][col] = 1
    V[2][col] = 1
    W[7][col] = 1
    col += 1
    # T4: A[1,1]B[1,2] -> C[1,2]
    U[4][col] = 1
    V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # A21 (6,7) * B11 (0,1,4,5) -> C21 (6,7) part?
    # A21 is A[2,0], A[2,1]. B11 is B[0,0], B[0,1], B[1,0], B[1,1].
    # Result C[2,0], C[2,1]. Indices C: 6,7? No, C[2,0] is 6, C[2,1] is 7.
    # T1: A[2,0]B[0,0] -> C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[6][col] = 1
    col += 1
    # T2: A[2,0]B[1,0] -> C[2,0]
    U[6][col] = 1
    V[4][col] = 1
    W[6][col] = 1
    col += 1
    # T3: A[2,1]B[0,1] -> C[2,1]
    U[7][col] = 1
    V[1][col] = 1
    W[7][col] = 1
    col += 1
    # T4: A[2,1]B[1,1] -> C[2,1]
    U[7][col] = 1
    V[5][col] = 1
    W[7][col] = 1
    col += 1
    
    # A21 (6,7) * B12 (2,3) -> C22 (8) part?
    # A21: A[2,0], A[2,1]. B12: B[0,2], B[1,2].
    # Result C[2,2]. Index C: 8.
    # T1: A[2,0]B[0,2] -> C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    # T2: A[2,1]B[1,2] -> C[2,2]
    U[7][col] = 1
    V[3][col] = 1
    W[8][col] = 1
    col += 1
    
    # A22 (8) * B21 (6,7) -> C21 (6,7) part?
    # A22: A[2,2]. B21: B[2,0], B[2,1].
    # Result C[2,0], C[2,1].
    # T1: A[2,2]B[2,0] -> C[2,0]
    U[8][col] = 1
    V[6][col] = 1
    W[6][col] = 1
    col += 1
    # T2: A[2,2]B[2,1] -> C[2,1]
    U[8][col] = 1
    V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # A22 (8) * B22 (8) -> C22 (8)
    # T1: A[2,2]B[2,2] -> C[2,2]
    U[8][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    # Part 2: 3x3 x 3x1 (4th column of B)
    # B indices: 9,10,11.
    # C indices: 9,10,11.
    # Standard multiplication.
    for i in range(3):
        for j in range(3):
            u_idx = i * 3 + j
            v_idx = j * 4 + 3
            w_idx = 3 * 3 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
