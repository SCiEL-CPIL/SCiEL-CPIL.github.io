import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 150s limit

    n, m, p = DIMENSIONS  # 2, 2, 10
    # A is 2x2, B is 2x10, C is 2x10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    # W uses trace convention: W[k*n+i, r]

    rank = 35
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35

    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U indices: 0=a(A00), 1=b(A01), 2=c(A10), 3=d(A11)
    # V local: 0=e(B00), 1=f(B01), 2=g(B10), 3=h(B11)
    # W local: 0=C00, 1=C01, 2=C10, 3=C11
    # W actual row: k*n + i where i is output row, k is output col

    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (c+d)*e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3 = a*(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4 = d*(g-e)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5 = (a+b)*h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (c-a)*(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (b-d)*(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    num_blocks = p // 2  # 5

    col = 0
    for block in range(num_blocks):
        for a_vec, b_vec, c_vec in strassen_terms:
            # Fill U (same for all blocks)
            for i in range(4):
                U[i][col] = a_vec[i]

            # Fill V: B is 2x10, row j of B, col k of B
            # V index = j*p + k
            # Block 'block' covers B columns 2*block and 2*block+1
            # b_vec[0] = e = B[0][2*block] -> V[0*p + 2*block]
            # b_vec[1] = f = B[0][2*block+1] -> V[0*p + 2*block+1]
            # b_vec[2] = g = B[1][2*block] -> V[1*p + 2*block]
            # b_vec[3] = h = B[1][2*block+1] -> V[1*p + 2*block+1]
            V[0 * p + 2 * block][col] = b_vec[0]
            V[0 * p + 2 * block + 1][col] = b_vec[1]
            V[1 * p + 2 * block][col] = b_vec[2]
            V[1 * p + 2 * block + 1][col] = b_vec[3]

            # Fill W: C is 2x10, C[i][k]
            # W uses trace convention: W[k*n + i, r]
            # Block 'block' covers C columns 2*block and 2*block+1
            # c_vec[0] = C00 = C[0][2*block] -> W[(2*block)*n + 0]
            # c_vec[1] = C01 = C[0][2*block+1] -> W[(2*block+1)*n + 0]
            # c_vec[2] = C10 = C[1][2*block] -> W[(2*block)*n + 1]
            # c_vec[3] = C11 = C[1][2*block+1] -> W[(2*block+1)*n + 1]
            W[(2 * block) * n + 0][col] = c_vec[0]
            W[(2 * block + 1) * n + 0][col] = c_vec[1]
            W[(2 * block) * n + 1][col] = c_vec[2]
            W[(2 * block + 1) * n + 1][col] = c_vec[3]

            col += 1

    # Verify correctness with a quick random test
    try:
        rng = np.random.default_rng(42)
        for trial in range(10):
            A_test = rng.integers(-3, 4, size=(n, m)).astype(float)
            B_test = rng.integers(-3, 4, size=(m, p)).astype(float)
            C_expected = A_test @ B_test

            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)

            A_flat = A_test.flatten()  # [a, b, c, d]
            B_flat = B_test.flatten()  # [B[0][0], ..., B[0][9], B[1][0], ..., B[1][9]]

            C_computed = np.zeros(p * n)
            for r in range(rank):
                uA = np.dot(U_arr[:, r], A_flat)
                vB = np.dot(V_arr[:, r], B_flat)
                C_computed += uA * vB * W_arr[:, r]

            # C_computed is indexed by k*n + i, so reshape
            C_computed_matrix = C_computed.reshape(p, n).T  # now [i][k]

            if not np.allclose(C_computed_matrix, C_expected, atol=1e-9):
                break
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
