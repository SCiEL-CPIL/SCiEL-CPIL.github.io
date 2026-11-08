import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 2x2 Strassen factors (rank 7)
    # u[r][k] where r in 0..3 (row*2+col), k in 0..6
    s_u = [
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1],
    ]
    s_v = [
        [1,  0,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  1, -1,  0,  1,  0,  1],
    ]
    s_w = [
        [1,  0,  0,  1, -1,  0,  1],
        [0,  0,  1,  0,  1,  0,  0],
        [0,  1,  0,  1,  0,  0,  0],
        [1, -1,  1,  0,  0,  1,  0],
    ]

    # 7 Strassen terms for 4x4 block decomposition
    # Each term: (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    # Block (br, bc) is a 2x2 sub-block
    terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def add_4x4_term(a_coeffs, b_coeffs, c_coeffs, start_col):
        for k in range(7):
            col_idx = start_col + k
            # Add U contributions
            for (a_br, a_bc), a_coeff in a_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = s_u[local_idx][k]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row][col_idx] += a_coeff * u_val
            # Add V contributions
            for (b_br, b_bc), b_coeff in b_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = s_v[local_idx][k]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row][col_idx] += b_coeff * v_val
            # Add W contributions
            for (c_br, c_bc), c_coeff in c_coeffs.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = s_w[local_idx][k]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row][col_idx] += c_coeff * w_val

    # Add 28 terms for the 4x4 core (7 Strassen terms * 4 block positions... wait, no)
    # Actually, the 4x4 matrix is decomposed into 2x2 blocks, and each of the 7 Strassen
    # multiplies operates on 2x2 blocks. So we have 7 terms total for the 4x4 core.
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        add_4x4_term(a_coeff, b_coeff, c_coeff, start_col)

    # Add border terms: 36 terms
    # Top-right 4x1: A[i,4] * B[4,k] -> C[i,k] for i in 0..3, k in 0..3 (16 terms)
    col = 28
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1

    # Bottom-left 1x4: A[4,j] * B[j,k] -> C[4,k]... wait, A is 4x5, so row 4 doesn't exist for A
    # A is 4x5 (rows 0-3, cols 0-4), B is 5x5 (rows 0-4, cols 0-4), C is 5x4 (rows 0-4, cols 0-3)
    # Border terms:
    # 1. A[i,4] * B[4,k] for i in 0..3, k in 0..3 -> C[i,k] (16 terms) - done above
    # 2. A[i,4] * B[j,4] for i in 0..3, j in 0..3 -> C[j,i]... no
    # Let me reconsider. C = A * B where A is 4x5, B is 5x5, C is 4x5? No, C is 5x4?
    # Task says: 4 by 5 matrix by 5 by 5 matrix. So A is 4x5, B is 5x5, C is 4x5.
    # But W shape is (p*n, rank) = (5*4, rank) = (20, rank). So C is 5x4? That's transposed.
    # Output factor follows W[k*n+i, r]. So W row index = k*4 + i where k is 0..4 (col of B / row of C?) 
    # Actually n=4, m=5, p=5. U shape (n*m, rank)=(20,rank), V shape (m*p,rank)=(25,rank), W shape (p*n,rank)=(20,rank).
    # A is n x m = 4x5, B is m x p = 5x5, C is n x p = 4x5.
    # But W is p x n = 5x4? That seems like C is stored as p x n = 5x4, which is C^T.
    # W[k*n+i, r] means row index = k*4+i, so k is 0..4 (5 values) and i is 0..3 (4 values).
    # So C is stored with k as the first index (0..4) and i as the second (0..3). C[k,i] = sum_j A[i,j]*B[j,k].
    # So C is 5x4 where C[k,i] = (A*B)^T[k,i] = (A*B)[i,k].
    
    # Border terms:
    # 1. A[i,4]*B[4,k] for i in 0..3, k in 0..3 -> C[k,i] (16 terms)
    # 2. A[i,4]*B[j,4] for i in 0..3, j in 0..3 -> C[4,i]... no, B[j,4] with j in 0..3
    #    A[i,4]*B[j,4] doesn't make sense for C[k,i]
    # Let me list all border terms:
    # C[k,i] = sum_{j=0}^{4} A[i,j]*B[j,k]
    # Core: j in 0..3, k in 0..3 -> handled by 4x4 Strassen (28 terms)
    # Border 1: j=4, k in 0..3, i in 0..3 -> A[i,4]*B[4,k] -> C[k,i] (16 terms)
    # Border 2: j in 0..3, k=4, i in 0..3 -> A[i,j]*B[j,4] -> C[4,i] (16 terms)
    # Border 3: j=4, k=4, i in 0..3 -> A[i,4]*B[4,4] -> C[4,i] (4 terms)
    # Total border: 16+16+4 = 36 terms. Total rank: 28+36=64.

    # Border 2: A[i,j]*B[j,4] -> C[4,i] for i in 0..3, j in 0..3
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Border 3: A[i,4]*B[4,4] -> C[4,i] for i in 0..3
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
