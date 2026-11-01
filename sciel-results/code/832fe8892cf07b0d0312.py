"""Matrix multiplication 2x3x6 with rank-minimal exact tensor decomposition.

Uses a block-structured decomposition:
- Split A into A1 (2x2) and a2 (2x1)
- Split B into B1 (2x6) and b2 (1x6)
- For each 2x2 block of B1 (there are 3 such blocks), use Strassen's rank-7 algorithm
- For the last column, use trivial rank-6 decomposition

Total rank: 3*7 + 6 = 27
"""

def construct():
    n, m, p = 2, 3, 6
    R = 27
    U = [[0] * R for _ in range(n * m)]  # 6 x 27
    V = [[0] * R for _ in range(m * p)]  # 18 x 27
    W = [[0] * R for _ in range(p * n)]  # 12 x 27

    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B22-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B22+B12)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Strassen coefficients: for each multiplication s in 0..6,
    # U_coeffs[s][i] for i in 0..3 (A indices: A11=0, A12=1, A21=2, A22=3)
    # V_coeffs[s][j] for j in 0..3 (B indices: B11=0, B12=1, B21=2, B22=3)
    # W_coeffs[s][c] for c in 0..3 (C indices: C11=0, C12=1, C21=2, C22=3)

    # U coefficients for each Strassen multiplication
    U_coeff = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]

    # V coefficients for each Strassen multiplication
    V_coeff = [
        [1, 0, 1, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 0, 1, -1], # M3: B21 - B22
        [0, 0, 0, 1],  # M4: B22
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B21
        [0, 1, 0, 1],  # M7: B22 + B12
    ]

    # W coefficients for each Strassen multiplication
    W_coeff = [
        [1, 0, 0, -1],  # M1: C11 + C22
        [0, 0, 1, -1],  # M2: C21 - C22
        [0, 1, 0, 1],   # M3: C12 + C22
        [1, 0, 1, 0],   # M4: C11 + C21
        [-1, 1, 0, 0],  # M5: -C11 + C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 1],   # M7: C11 + C22
    ]

    # Part 1: A1 * B1 using Strassen for each 2x2 block of B1
    # A1 = A[:, 0:2] (columns 0,1), B1 = B[0:2, :] (rows 0,1)
    # For each block b in 0,1,2 (columns 2b, 2b+1 of B1):
    for b in range(3):
        k0 = 2 * b  # first column of this block
        k1 = 2 * b + 1  # second column of this block
        r_offset = b * 7  # rank offset for this block

        for s in range(7):
            r = r_offset + s

            # U: A1 indices (i=0,1; j=0,1) -> U[i*3 + j]
            # A11 = A[0,0] -> U[0*3+0] = U[0]
            # A12 = A[0,1] -> U[0*3+1] = U[1]
            # A21 = A[1,0] -> U[1*3+0] = U[3]
            # A22 = A[1,1] -> U[1*3+1] = U[4]
            u_indices = [0, 1, 3, 4]  # A11, A12, A21, A22
            for idx, u_val in enumerate(U_coeff[s]):
                if u_val != 0:
                    U[u_indices[idx]][r] = u_val

            # V: B1[j, k] where j=0,1 and k=k0,k1
            # B11 = B[0,k0] -> V[0*6+k0] = V[k0]
            # B12 = B[0,k1] -> V[0*6+k1] = V[k1]
            # B21 = B[1,k0] -> V[1*6+k0] = V[6+k0]
            # B22 = B[1,k1] -> V[1*6+k1] = V[6+k1]
            v_indices = [k0, k1, 6 + k0, 6 + k1]  # B11, B12, B21, B22
            for idx, v_val in enumerate(V_coeff[s]):
                if v_val != 0:
                    V[v_indices[idx]][r] = v_val

            # W: C[i, k] where i=0,1 and k=k0,k1
            # C index = k*n + i = k*2 + i
            # C11 = C[0,k0] -> W[k0*2+0] = W[2*k0]
            # C12 = C[0,k1] -> W[k1*2+0] = W[2*k1]
            # C21 = C[1,k0] -> W[k0*2+1] = W[2*k0+1]
            # C22 = C[1,k1] -> W[k1*2+1] = W[2*k1+1]
            w_indices = [2 * k0, 2 * k1, 2 * k0 + 1, 2 * k1 + 1]  # C11, C12, C21, C22
            for idx, w_val in enumerate(W_coeff[s]):
                if w_val != 0:
                    W[w_indices[idx]][r] = w_val

    # Part 2: a2 * b2 (trivial)
    # a2 = A[:, 2], b2 = B[2, :]
    # For each i in 0,1 and k in 0..5:
    r_offset2 = 21
    for i in range(2):
        for k in range(6):
            r = r_offset2 + i * 6 + k
            U[i * 3 + 2][r] = 1  # A[i, 2]
            V[2 * 6 + k][r] = 1  # B[2, k]
            W[k * 2 + i][r] = 1  # C[i, k]

    return {"U": U, "V": V, "W": W}
