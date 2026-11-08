import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Build Strassen's rank-7 decomposition for 2x2x2
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A * B where C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    #
    # U indices: (i, j) for A[i,j], flattened as i*2+j
    # V indices: (j, k) for B[j,k], flattened as j*2+k
    # W indices: (k, i) for C[i,k], flattened as k*2+i
    #
    # Strassen products:
    # M1: (A11+A22)(B11+B22) -> C11+=1, C22+=1
    # M2: (A21+A22)B11      -> C21+=1, C22-=1
    # M3: A11(B21-B22)      -> C12+=1, C22+=1
    # M4: A22(B12-B21)      -> C11+=1, C21+=1
    # M5: (A11+A12)B22      -> C11-=1, C12+=1
    # M6: (A21-A11)(B11+B12)-> C22+=1
    # M7: (A12-A22)(B21+B22)-> C11+=1

    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from (i,j) to coeff
    # V_coeffs: dict from (j,k) to coeff
    # W_coeffs: dict from (k,i) to coeff

    strassen_terms = [
        # M1: (A11+A22)(B11+B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A21+A22)B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # M3: A11(B21-B22)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: A22(B12-B21)
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (0,1):1}),
        # M5: (A11+A12)B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A21-A11)(B11+B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A12-A22)(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # Now build the full 4x4x6 decomposition
    # A is 4x4: blocks A[bi][bj] for bi in {0,1}, bj in {0,1}
    # B is 4x6: blocks B[bj][bk] for bj in {0,1}, bk in {0,1,2}
    # C is 4x6: blocks C[bi][bk] for bi in {0,1}, bk in {0,1,2}
    #
    # For each (bi, bj, bk), we compute C[bi][bk] += A[bi][bj] * B[bj][bk]
    # Each such product is a 2x2 * 2x2 = 2x2 matrix product
    # Using Strassen, each contributes 7 rank-1 terms
    # Total: 2*2*3*7 = 84 terms

    total_rank = 0
    U_list = []
    V_list = []
    W_list = []

    for bi in range(2):      # row block of A/C
        for bj in range(2):  # col block of A / row block of B
            for bk in range(3):  # col block of B/C
                for (u_coeff, v_coeff, w_coeff) in strassen_terms:
                    # Build U row: 4*4 = 16 entries, index = i*4+j
                    # A[bi][bj] block: A[i,j] where i = bi*2+si, j = bj*2+sj
                    u_row = [0] * 16
                    for (si, sj), coeff in u_coeff.items():
                        i = bi * 2 + si
                        j = bj * 2 + sj
                        u_row[i * 4 + j] = coeff

                    # Build V row: 4*6 = 24 entries, index = j*6+k
                    # B[bj][bk] block: B[j,k] where j = bj*2+sj, k = bk*2+sk
                    v_row = [0] * 24
                    for (sj, sk), coeff in v_coeff.items():
                        j = bj * 2 + sj
                        k = bk * 2 + sk
                        v_row[j * 6 + k] = coeff

                    # Build W row: 6*4 = 24 entries, index = k*4+i
                    # C[bi][bk] block: C[i,k] where i = bi*2+si, k = bk*2+sk
                    w_row = [0] * 24
                    for (sk, si), coeff in w_coeff.items():
                        k = bk * 2 + sk
                        i = bi * 2 + si
                        w_row[k * 4 + i] = coeff

                    U_list.append(u_row)
                    V_list.append(v_row)
                    W_list.append(w_row)
                    total_rank += 1

    # Verify the decomposition
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)

    # Target tensor: T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,6)
    # Check: sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r] == 1

    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k])  # This isn't right, need element-wise
                # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                error = max(error, abs(val - 1.0))

    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U_fallback = [[0] * rank for _ in range(n * m)]
        V_fallback = [[0] * rank for _ in range(m * p)]
        W_fallback = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j][col] = 1
                    V_fallback[j * p + k][col] = 1
                    W_fallback[k * n + i][col] = 1
                    col += 1
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

    # Convert to lists of integers/half-integers
    U_out = []
    for row in U_list:
        U_out.append([int(x) if abs(x - round(x)) < 1e-9 else (2*int(2*x))/2 for x in row])
    V_out = []
    for row in V_list:
        V_out.append([int(x) if abs(x - round(x)) < 1e-9 else (2*int(2*x))/2 for x in row])
    W_out = []
    for row in W_list:
        W_out.append([int(x) if abs(x - round(x)) < 1e-9 else (2*int(2*x))/2 for x in row])

    return {"U": U_out, "V": V_out, "W": W_out}
