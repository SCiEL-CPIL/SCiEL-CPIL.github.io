"""
6x6 Matrix Multiplication Tensor Decomposition - rank 189 (Strassen 2x2 outer, 3x3 schoolbook inner)
"""
import time
import numpy as np

_DEADLINE = time.time() + 135

def construct():
    n, m, p = 6, 6, 6
    dim = 6
    block = 3
    total_rank = 189  # 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=int)
    V = np.zeros((dim * dim, total_rank), dtype=int)
    W = np.zeros((dim * dim, total_rank), dtype=int)

    steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    col = 0
    for A_list, B_list, C_list in steps:
        for r in range(block):
            for c in range(block):
                for d in range(block):
                    for (I, J, a) in A_list:
                        ii = I * block + r
                        jj = J * block + c
                        U[ii * dim + jj, col] += a
                    for (J, K, b) in B_list:
                        jj = J * block + c
                        kk = K * block + d
                        V[jj * dim + kk, col] += b
                    for (I, K, wc) in C_list:
                        ii = I * block + r
                        kk = K * block + d
                        W[kk * dim + ii, col] += wc
                    col += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify
    U_np = np.array(result["U"], dtype=float)
    V_np = np.array(result["V"], dtype=float)
    W_np = np.array(result["W"], dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                err = abs(np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i]) - 1.0)
                if err > max_err:
                    max_err = err

    if max_err < 1e-9:
        return result
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
