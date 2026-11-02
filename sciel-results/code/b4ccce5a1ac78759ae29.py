"""Improved program for the suite task."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition using block partitioning.
    # This is a valid exact decomposition.
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Block 1: Top-left 2x2 multiplication (Strassen, Rank 7)
    # U indices: 0,1,3,4 (A00, A01, A10, A11)
    # V indices: 0,1,3,4 (B00, B01, B10, B11)
    # W indices: 0,3,1,4 (C00, C01, C10, C11)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: Cross terms (Rank 19)
    
    # 2a: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # U: 2, 5. V: 6, 7. W: 0, 3, 1, 4
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # 2b: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # U: 0,1,3,4. V: 2, 5. W: 6, 7
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # 2c: A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    # U: 2, 5. V: 8. W: 6, 7
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # 2d: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # U: 6, 7. V: 0,1,3,4. W: 2, 5
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # 2e: A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    # U: 8. V: 6, 7. W: 2, 5
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # 2f: A[2, 0:2] * B[0:2, 2] -> C[2, 2]
    # U: 6, 7. V: 2, 5. W: 8
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # 2g: A[2, 2] * B[2, 2] -> C[2, 2]
    # U: 8. V: 8. W: 8
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
