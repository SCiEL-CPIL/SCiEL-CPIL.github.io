"""
Solution for 4x4 matrix multiplication with minimal rank 49.

Uses a 2-level Strassen decomposition (Kronecker product of two 2x2 Strassen decompositions).
The rank is 7 * 7 = 49.
"""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 4, 4, 4
    rank = 49

    # --- 2x2 Strassen Coefficients ---
    U_elem = [
        [1, 0, 0, 1],    # t=0: A00 + A11
        [0, 0, 1, 1],    # t=1: A10 + A11
        [1, 0, 0, 0],    # t=2: A00
        [0, 0, 0, 1],    # t=3: A11
        [1, 1, 0, 0],    # t=4: A00 + A01
        [-1, 0, 1, 0],   # t=5: A10 - A00
        [0, 1, 0, -1]    # t=6: A01 - A11
    ]
    V_elem = [
        [1, 0, 0, 1],    # t=0: B00 + B11
        [1, 0, 0, 0],    # t=1: B00
        [0, 1, 0, -1],   # t=2: B01 - B11
        [-1, 0, 1, 0],   # t=3: B10 - B00
        [0, 0, 0, 1],    # t=4: B11
        [1, 1, 0, 0],    # t=5: B00 + B01
        [0, 0, 1, 1]     # t=6: B10 + B11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0: C00+=M, C11+=M
        [[0, 0], [1, -1]],  # t=1: C10+=M, C11-=M
        [[0, 1], [0, 1]],   # t=2: C01+=M, C11+=M
        [[1, 0], [1, 0]],   # t=3: C00+=M, C10+=M
        [[-1, 1], [0, 0]],  # t=4: C00-=M, C01+=M
        [[0, 0], [0, 1]],   # t=5: C11+=M
        [[1, 0], [0, 0]]    # t=6: C00+=M
    ]

    # --- Block Level Strassen Coefficients ---
    A_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    B_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    C_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Initialize U, V, W matrices
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (bi * 2 + ei) * 4 + (bj * 2 + ej)
                        U[row_a, col] += c_a * U_elem[t][ei * 2 + ej]
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (bj * 2 + ej) * 4 + (bk * 2 + ek)
                        V[row_b, col] += c_b * V_elem[t][ej * 2 + ek]
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        row_c = (bk * 2 + ek) * 4 + (bi * 2 + ei)
                        W[row_c, col] += c_c * W_elem[t][ei][ek]

    def verify(U, V, W, n, m, p, rank):
        if time.time() - start_time > deadline - 10:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Build the multiplication tensor T where T[i*m+j, j*p+k, k*n+i] = 1
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        D = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        return np.allclose(T, D, atol=1e-9)

    if verify(U, V, W, n, m, p, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: trivial rank-64 decomposition
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=int)
        V_fb = np.zeros((16, rank_fb), dtype=int)
        W_fb = np.zeros((16, rank_fb), dtype=int)
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col_fb] = 1
                    V_fb[j*4+k, col_fb] = 1
                    W_fb[k*4+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
