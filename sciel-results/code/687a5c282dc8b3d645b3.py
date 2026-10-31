import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = 4, 4, 4
    rank = 49

    # Define the 7 Strassen multiplications for 2x2 matrices
    # Each multiplication s has:
    #   M_s = (sum_{i,j} a_s[i][j] * A[i][j]) * (sum_{j,k} b_s[j][k] * B[j][k])
    #   C[i][k] += c_s[i][k] * M_s
    #
    # Using 0-indexed: A[i][j], B[j][k], C[i][k] with i,j,k in {0,1}
    #
    # Standard Strassen (0-indexed):
    # s=0: M0 = (A00+A11)(B00+B11), C00+=M0, C11+=M0
    # s=1: M1 = (A10+A11)B00, C10+=M1, C11-=M1
    # s=2: M2 = A00(B01-B11), C01+=M2, C11+=M2
    # s=3: M3 = A11(B10-B00), C00+=M3, C10+=M3
    # s=4: M4 = (A00+A01)B11, C00-=M4, C01+=M4
    # s=5: M5 = (A10-A00)(B00+B01), C11+=M5
    # s=6: M6 = (A01-A11)(B10+B11), C00+=M6

    # a_coeffs[s][i][j]: coefficient for A[i][j] in s-th multiplication
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, 1]],   # s=1
        [[1, 0], [0, 0]],   # s=2
        [[0, 0], [0, 1]],   # s=3
        [[1, 1], [0, 0]],   # s=4
        [[-1, 0], [1, 0]],  # s=5
        [[0, 1], [0, -1]],  # s=6
    ], dtype=float)

    # b_coeffs[s][j][k]: coefficient for B[j][k] in s-th multiplication
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[1, 0], [0, 0]],   # s=1
        [[0, 1], [0, -1]],  # s=2
        [[-1, 0], [1, 0]],  # s=3
        [[0, 0], [0, 1]],   # s=4
        [[1, 1], [0, 0]],   # s=5
        [[0, 0], [1, 1]],   # s=6
    ], dtype=float)

    # c_coeffs[s][i][k]: coefficient for C[i][k] in s-th multiplication
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, -1]],  # s=1
        [[0, 1], [0, 1]],   # s=2
        [[1, 0], [1, 0]],   # s=3
        [[-1, 1], [0, 0]],  # s=4
        [[0, 0], [0, 1]],   # s=5
        [[1, 0], [0, 0]],   # s=6
    ], dtype=float)

    # Build the 49-rank decomposition using Kronecker product structure
    # The 4x4 matrix multiplication tensor can be decomposed as:
    # T_4 = sum_{s=0}^{6} sum_{t=0}^{6} u_{s,t} ⊗ v_{s,t} ⊗ w_{s,t}
    # where:
    #   u_{s,t} = a_coeffs[s] ⊗ a_coeffs[t]  (shape 4x4, flattened to 16)
    #   v_{s,t} = b_coeffs[s] ⊗ b_coeffs[t]  (shape 4x4, flattened to 16)
    #   w_{s,t} = c_coeffs[s] ⊗ c_coeffs[t]  (shape 4x4, flattened to 16)
    #
    # Indexing conventions:
    #   U[i*m+j, col] = U[i*4+j, col] for A[i][j]
    #   V[j*p+k, col] = V[j*4+k, col] for B[j][k]
    #   W[k*n+i, col] = W[k*4+i, col] for C[i][k] (trace convention)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for s in range(7):
        for t in range(7):
            col = 7 * s + t

            # Compute Kronecker products
            # a_coeffs[s] is 2x2, a_coeffs[t] is 2x2
            # kron(a_coeffs[s], a_coeffs[t]) gives 4x4 matrix
            # Row index i*2+i2, col index j*2+j2 corresponds to A[i][j] where i=2*I+i2, j=2*J+j2
            
            # For U: A index is i*4+j where i = 2*I+i2, j = 2*J+j2
            # u_{s,t}[i][j] = a_coeffs[s][I][J] * a_coeffs[t][i2][j2]
            for I in range(2):
                for J in range(2):
                    a_s_val = a_coeffs[s][I][J]
                    if a_s_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            a_t_val = a_coeffs[t][i2][j2]
                            if a_t_val == 0:
                                continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U[i * 4 + j][col] = int(a_s_val * a_t_val)

            # For V: B index is j*4+k where j = 2*J+j2, k = 2*K+k2
            # v_{s,t}[j][k] = b_coeffs[s][J][K] * b_coeffs[t][j2][k2]
            for J in range(2):
                for K in range(2):
                    b_s_val = b_coeffs[s][J][K]
                    if b_s_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            b_t_val = b_coeffs[t][j2][k2]
                            if b_t_val == 0:
                                continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V[j * 4 + k][col] = int(b_s_val * b_t_val)

            # For W: C index is k*4+i (trace convention) where k = 2*K+k2, i = 2*I+i2
            # w_{s,t}[i][k] = c_coeffs[s][I][K] * c_coeffs[t][i2][k2]
            # W[k*4+i, col] = w_{s,t}[i][k]
            for I in range(2):
                for K in range(2):
                    c_s_val = c_coeffs[s][I][K]
                    if c_s_val == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            c_t_val = c_coeffs[t][i2][k2]
                            if c_t_val == 0:
                                continue
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W[k * 4 + i][col] = int(c_s_val * c_t_val)

            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
