import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Initialize factor matrices
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2
    # Coefficients for A, B, C linear forms
    # a_coeffs[s][i][j] is coeff of A[i][j] in M_s
    # b_coeffs[s][j][k] is coeff of B[j][k] in M_s
    # c_coeffs[s][i][k] is coeff of M_s in C[i][k]
    
    a_coeffs = [
        [[1, 0], [0, 1]],  # M0: (A00+A11)
        [[0, 0], [1, 1]],  # M1: (A10+A11)
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: (A00+A01)
        [[-1, 0], [1, 0]], # M5: (A10-A00)
        [[0, 1], [0, -1]]  # M6: (A01-A11)
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],  # M0: (B00+B11)
        [[1, 0], [0, 0]],  # M1: B00
        [[0, 1], [0, -1]], # M2: (B01-B11)
        [[-1, 0], [1, 0]], # M3: (B10-B00)
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: (B00+B01)
        [[0, 0], [1, 1]]   # M6: (B10+B11)
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],  # M0: C00+=M0, C11+=M0
        [[0, 0], [1, -1]], # M1: C10+=M1, C11-=M1
        [[0, 1], [0, 1]],  # M2: C01+=M2, C11+=M2
        [[1, 0], [1, 0]],  # M3: C00+=M3, C10+=M3
        [[-1, 1], [0, 0]], # M4: C00-=M4, C01+=M4
        [[0, 0], [0, 1]],  # M5: C11+=M5
        [[1, 0], [0, 0]]   # M6: C00+=M6
    ]

    # Construct 4x4 decomposition using Kronecker product logic
    # Outer Strassen (s) operates on 2x2 blocks of 2x2 matrices
    # Inner Strassen (t) operates on the 2x2 elements within those blocks
    
    for s in range(7):
        for t in range(7):
            rank_idx = s * 7 + t
            
            # U: rows are (i, j) for A
            # i = 2*S + s2, j = 2*T + t2
            # U[i*4+j, rank_idx] = a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            for S in range(2):
                for T in range(2):
                    coeff_outer = a_coeffs[s][S][T]
                    if coeff_outer == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            coeff_inner = a_coeffs[t][s2][t2]
                            if coeff_inner == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][rank_idx] = coeff_outer * coeff_inner

            # V: rows are (j, k) for B
            # j = 2*T + t2, k = 2*K + k2
            # V[j*4+k, rank_idx] = b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            for T in range(2):
                for K in range(2):
                    coeff_outer = b_coeffs[s][T][K]
                    if coeff_outer == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            coeff_inner = b_coeffs[t][t2][k2]
                            if coeff_inner == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][rank_idx] = coeff_outer * coeff_inner

            # W: rows are (k, i) for C (transposed index convention W[k*n+i])
            # C[i, k] corresponds to W[k*4+i]
            # i = 2*S + s2, k = 2*K + k2
            # W[k*4+i, rank_idx] = c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            for S in range(2):
                for K in range(2):
                    coeff_outer = c_coeffs[s][S][K]
                    if coeff_outer == 0:
                        continue
                    for s2 in range(2):
                        for k2 in range(2):
                            coeff_inner = c_coeffs[t][s2][k2]
                            if coeff_inner == 0:
                                continue
                            i = 2 * S + s2
                            k = 2 * K + k2
                            W[k * 4 + i][rank_idx] = coeff_outer * coeff_inner

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
