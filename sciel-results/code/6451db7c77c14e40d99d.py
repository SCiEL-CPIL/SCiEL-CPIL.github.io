"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is p x n

    def E(i, j, rows, cols):
        M = np.zeros((rows, cols))
        M[i, j] = 1.0
        return M

    def eA(i, j): return E(i, j, 3, 3)
    def eB(j, k): return E(j, k, 3, 4)
    def eC(i, k): return E(i, k, 4, 3)

    terms = []

    def add_term(U_mat, V_mat, W_mat):
        terms.append((U_mat, V_mat, W_mat))

    a00, a01, a10, a11 = eA(0,0), eA(0,1), eA(1,0), eA(1,1)
    b00, b01, b10, b11 = eB(0,0), eB(0,1), eB(1,0), eB(1,1)
    c00, c01, c10, c11 = eC(0,0), eC(0,1), eC(1,0), eC(1,1)
    b02, b03, b12, b13 = eB(0,2), eB(0,3), eB(1,2), eB(1,3)
    c02, c03, c12, c13 = eC(0,2), eC(0,3), eC(1,2), eC(1,3)
    c20, c21, c22, c23 = eC(2,0), eC(2,1), eC(2,2), eC(2,3)

    # C[0:2,0:2] = A[0:2,0:2]@B[0:2,0:2] (Strassen, rank 7) + A[0:2,2]@B[2,0:2] (schoolbook, rank 4)
    add_term(a00+a11, b00+b11, c00+c11-c10-c01)  # M1
    add_term(a10+a11, b00, c10+c11)               # M2
    add_term(a00, b01-b11, c01+c11)               # M3
    add_term(a11, b10-b00, c00+c10)               # M4
    add_term(a00+a01, b11, -c00+c01)              # M5
    add_term(a10-a00, b00+b01, c11)               # M6
    add_term(a01-a11, b10+b11, c00-c11)           # M7
    add_term(eA(0,2), eB(2,0), c00)
    add_term(eA(0,2), eB(2,1), c01)
    add_term(eA(1,2), eB(2,0), c10)
    add_term(eA(1,2), eB(2,1), c11)

    # C[0:2,2:4] = A[0:2,0:2]@B[0:2,2:4] (Strassen, rank 7) + A[0:2,2]@B[2,2:4] (schoolbook, rank 4)
    add_term(a00+a11, b02+b13, c02+c13-c12-c03)  # M1'
    add_term(a10+a11, b02, c12+c13)               # M2'
    add_term(a00, b03-b13, c03+c13)               # M3'
    add_term(a11, b12-b02, c02+c12)               # M4'
    add_term(a00+a01, b13, -c02+c03)              # M5'
    add_term(a10-a00, b02+b03, c13)               # M6'
    add_term(a01-a11, b12+b13, c02-c13)           # M7'
    add_term(eA(0,2), eB(2,2), c02)
    add_term(eA(0,2), eB(2,3), c03)
    add_term(eA(1,2), eB(2,2), c12)
    add_term(eA(1,2), eB(2,3), c13)

    # C[2,0:2] = A[2,0:2]@B[0:2,0:2] (schoolbook, rank 4) + A[2,2]@B[2,0:2] (schoolbook, rank 2)
    add_term(eA(2,0), b00, c20)
    add_term(eA(2,0), b01, c21)
    add_term(eA(2,1), b10, c20)
    add_term(eA(2,1), b11, c21)
    add_term(eA(2,2), eB(2,0), c20)
    add_term(eA(2,2), eB(2,1), c21)

    # C[2,2:4] = A[2,0:2]@B[0:2,2:4] (schoolbook, rank 4) + A[2,2]@B[2,2:4] (schoolbook, rank 2)
    add_term(eA(2,0), b02, c22)
    add_term(eA(2,0), b03, c23)
    add_term(eA(2,1), b12, c22)
    add_term(eA(2,1), b13, c23)
    add_term(eA(2,2), eB(2,2), c22)
    add_term(eA(2,2), eB(2,3), c23)

    rank = len(terms)

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for c, (U_mat, V_mat, W_mat) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                U[i * m + j, c] = U_mat[i, j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, c] = V_mat[j, k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = W_mat[k, i]

    def verify_decomposition(U_arr, V_arr, W_arr, num_tests=100):
        np.random.seed(42)
        U_flat = U_arr.T
        V_flat = V_arr.T
        W_flat = W_arr.T
        for t in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            u_vals = U_flat @ A_flat
            v_vals = V_flat @ B_flat
            coeffs = u_vals * v_vals
            C_recon = (coeffs[:, None] * W_flat).sum(axis=0).reshape(p, n).T
            if not np.allclose(C_recon, C_true, atol=1e-10):
                return False
        return True

    valid = verify_decomposition(U, V, W)

    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]

    for i in range(n * m):
        for c in range(rank):
            val = U[i, c]
            U_list[i][c] = int(round(val)) if abs(val - round(val)) < 1e-10 else (0 if abs(val) < 1e-15 else val)

    for i in range(m * p):
        for c in range(rank):
            val = V[i, c]
            V_list[i][c] = int(round(val)) if abs(val - round(val)) < 1e-10 else (0 if abs(val) < 1e-15 else val)

    for i in range(p * n):
        for c in range(rank):
            val = W[i, c]
            W_list[i][c] = int(round(val)) if abs(val - round(val)) < 1e-10 else (0 if abs(val) < 1e-15 else val)

    return {"U": U_list, "V": V_list, "W": W_list}
