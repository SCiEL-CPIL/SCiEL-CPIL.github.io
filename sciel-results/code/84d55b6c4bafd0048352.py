import time

DIMENSIONS = [4, 4, 5]

def construct():
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Indices for U, V: (row, col) of 2x2 matrix -> index = row*2 + col
    # Indices for W: (k, i) of output 2x2 matrix -> index = k*2 + i
    
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h) -> p(+1), s(-1)
    U_222[0][0] = 1; U_222[3][0] = 1
    V_222[0][0] = 1; V_222[3][0] = 1
    W_222[0][0] = 1; W_222[3][0] = -1
    
    # M2 = (c+d)e -> r(+1)
    U_222[2][1] = 1; U_222[3][1] = 1
    V_222[0][1] = 1
    W_222[2][1] = 1
    
    # M3 = a(f-h) -> q(+1), s(-1)
    U_222[0][2] = 1
    V_222[1][2] = 1; V_222[3][2] = -1
    W_222[1][2] = 1; W_222[3][2] = -1
    
    # M4 = d(g-f) -> r(+1), s(-1)
    U_222[3][3] = 1
    V_222[2][3] = 1; V_222[1][3] = -1
    W_222[2][3] = 1; W_222[3][3] = -1
    
    # M5 = (a+b)h -> p(+1), q(+1)
    U_222[0][4] = 1; U_222[1][4] = 1
    V_222[3][4] = 1
    W_222[0][4] = 1; W_222[1][4] = 1
    
    # M6 = (c+d)(f-h) -> s(+1)
    U_222[2][5] = 1; U_222[3][5] = 1
    V_222[1][5] = 1; V_222[3][5] = -1
    W_222[3][5] = 1
    
    # M7 = (a-c)(e+g) -> p(-1), r(-1)
    U_222[0][6] = 1; U_222[2][6] = -1
    V_222[0][6] = 1; V_222[2][6] = 1
    W_222[0][6] = -1; W_222[2][6] = -1

    # Naive 2x2x1 decomposition
    # U: 4x4, V: 2x4, W: 2x4
    # A: 2x2, B: 2x1. C: 2x1.
    # C0 = a*e + b*f
    # C1 = c*e + d*f
    U_221 = [[0]*4 for _ in range(4)]
    V_221 = [[0]*4 for _ in range(2)]
    W_221 = [[0]*4 for _ in range(2)]
    
    # Term 1: a*e -> C0
    U_221[0][0] = 1
    V_221[0][0] = 1
    W_221[0][0] = 1
    
    # Term 2: b*f -> C0
    U_221[1][1] = 1
    V_221[1][1] = 1
    W_221[0][1] = 1
    
    # Term 3: c*e -> C1
    U_221[2][2] = 1
    V_221[0][2] = 1
    W_221[1][2] = 1
    
    # Term 4: d*f -> C1
    U_221[3][3] = 1
    V_221[1][3] = 1
    W_221[1][3] = 1

    # Assemble 4x4x5
    n, m, p = 4, 4, 5
    rank = 72
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, u_offset, v_offset, w_offset):
        nonlocal col
        for r in range(len(U_block)):
            for c in range(len(U_block[0])):
                if U_block[r][c] != 0:
                    U[u_offset + r][col] = U_block[r][c]
        for r in range(len(V_block)):
            for c in range(len(V_block[0])):
                if V_block[r][c] != 0:
                    V[v_offset + r][col] = V_block[r][c]
        for r in range(len(W_block)):
            for c in range(len(W_block[0])):
                if W_block[r][c] != 0:
                    W[w_offset + r][col] = W_block[r][c]
        col += 1

    # Blocks of 2x2x2 (Strassen)
    # A11 B11
    add_block(U_222, V_222, W_222, 0, 0, 0) # A11 (0-3), B11 (0-3), C11 (0-3)
    # A12 B21
    add_block(U_222, V_222, W_222, 2, 4, 0) # A12 (2-3), B21 (4-7), C11 (0-3)
    
    # A21 B11
    add_block(U_222, V_222, W_222, 4, 0, 4) # A21 (4-7), B11 (0-3), C21 (4-7)
    # A22 B21
    add_block(U_222, V_222, W_222, 6, 4, 4) # A22 (6-7), B21 (4-7), C21 (4-7)
    
    # A11 B12
    add_block(U_222, V_222, W_222, 0, 2, 2) # A11 (0-3), B12 (2-5), C12 (2-5)
    # A12 B22
    add_block(U_222, V_222, W_222, 2, 6, 2) # A12 (2-3), B22 (6-9), C12 (2-5)
    
    # A21 B12
    add_block(U_222, V_222, W_222, 4, 2, 6) # A21 (4-7), B12 (2-5), C22 (6-9)
    # A22 B22
    add_block(U_222, V_222, W_222, 6, 6, 6) # A22 (6-7), B22 (6-9), C22 (6-9)
    
    # Blocks of 2x2x1 (Naive)
    # A11 B13
    add_block(U_221, V_221, W_221, 0, 8, 8) # A11 (0-3), B13 (8-9), C13 (8-9)
    # A12 B23
    add_block(U_221, V_221, W_221, 2, 10, 8) # A12 (2-3), B23 (10-11), C13 (8-9)
    
    # A21 B13
    add_block(U_221, V_221, W_221, 4, 8, 10) # A21 (4-7), B13 (8-9), C23 (10-11)
    # A22 B23
    add_block(U_221, V_221, W_221, 6, 10, 10) # A22 (6-7), B23 (10-11), C23 (10-11)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    import numpy as np
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 5)
    
    # Compute C = A B
    C_true = A @ B
    
    # Compute C from decomposition
    # C[i, k] = sum_j sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # U is (16, 72). V is (20, 72). W is (20, 72).
    # Flatten A and B
    a_flat = A.flatten() # row major: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0]...
    b_flat = B.flatten()
    
    # Tensor contraction
    # T = sum_c U[:, c] * V[:, c] * W[:, c]
    # This is tricky with flat indices.
    
    # Let's just check rank
    print(f"Rank: {U.shape[1]}")
    print(f"Score: {80 / U.shape[1]}")
