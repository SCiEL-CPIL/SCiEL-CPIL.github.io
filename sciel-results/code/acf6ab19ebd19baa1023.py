import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = 3, 4, 6

    def check_exact(U, V, W):
        R = U.shape[1]
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Verify Strassen numerically
    def strassen_works():
        np.random.seed(123)
        for trial in range(10):
            A = np.random.randn(2, 2)
            B = np.random.randn(2, 2)
            C_exp = A @ B
            a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
            e, f, g, h = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
            m1 = (a + d) * (e + h)
            m2 = (c + d) * e
            m3 = a * (f - g)
            m4 = d * (g - h)
            m5 = (a + b) * h
            m6 = (c - d) * (e + f)
            m7 = (a - c) * (d + h)
            C00 = m1 + m4 - m5 + m7
            C01 = m3 + m5
            C10 = m2 + m4
            C11 = m1 - m2 + m3 + m6
            C_test = np.array([[C00, C01], [C10, C11]])
            if not np.allclose(C_test, C_exp, atol=1e-10):
                return False
        return True

    use_strassen = strassen_works()

    # Block structure
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 6)]

    # Compute total rank
    total_rank = 0
    for i_s, i_e in i_blocks:
        for j_s, j_e in j_blocks:
            for k_s, k_e in k_blocks:
                is_, js_, ks_ = i_e - i_s, j_e - j_s, k_e - k_s
                if is_ == 2 and js_ == 2 and ks_ == 2 and use_strassen:
                    total_rank += 7
                else:
                    total_rank += is_ * js_ * ks_

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0

    for i_s, i_e in i_blocks:
        for j_s, j_e in j_blocks:
            for k_s, k_e in k_blocks:
                is_, js_, ks_ = i_e - i_s, j_e - j_s, k_e - k_s
                if is_ == 2 and js_ == 2 and ks_ == 2 and use_strassen:
                    # Strassen rank 7 for 2x2 * 2x2 -> 2x2
                    # m1 = (a+d)(e+h)
                    U[i_s * m + j_s, col] = 1
                    U[(i_s + 1) * m + (j_s + 1), col] = 1
                    V[j_s * p + k_s, col] = 1
                    V[(j_s + 1) * p + (k_s + 1), col] = 1
                    W[k_s * n + i_s, col] = 1
                    W[(k_s + 1) * n + (i_s + 1), col] = 1
                    col += 1
                    # m2 = (c+d)e
                    U[(i_s + 1) * m + j_s, col] = 1
                    U[(i_s + 1) * m + (j_s + 1), col] = 1
                    V[j_s * p + k_s, col] = 1
                    W[k_s * n + (i_s + 1), col] = 1
                    W[(k_s + 1) * n + (i_s + 1), col] = -1
                    col += 1
                    # m3 = a(f-g)
                    U[i_s * m + j_s, col] = 1
                    V[j_s * p + (k_s + 1), col] = 1
                    V[(j_s + 1) * p + k_s, col] = -1
                    W[(k_s + 1) * n + i_s, col] = 1
                    W[(k_s + 1) * n + (i_s + 1), col] = 1
                    col += 1
                    # m4 = d(g-h)
                    U[(i_s + 1) * m + (j_s + 1), col] = 1
                    V[(j_s + 1) * p + k_s, col] = 1
                    V[(j_s + 1) * p + (k_s + 1), col] = -1
                    W[k_s * n + (i_s + 1), col] = 1
                    W[k_s * n + i_s, col] = 1
                    col += 1
                    # m5 = (a+b)h
                    U[i_s * m + j_s, col] = 1
                    U[i_s * m + (j_s + 1), col] = 1
                    V[(j_s + 1) * p + (k_s + 1), col] = 1
                    W[k_s * n + i_s, col] = -1
                    W[(k_s + 1) * n + i_s, col] = 1
                    col += 1
                    # m6 = (c-d)(e+f)
                    U[(i_s + 1) * m + j_s, col] = 1
                    U[(i_s + 1) * m + (j_s + 1), col] = -1
                    V[j_s * p + k_s, col] = 1
                    V[j_s * p + (k_s + 1), col] = 1
                    W[(k_s + 1) * n + (i_s + 1), col] = 1
                    col += 1
                    # m7 = (a-c)(d+h)
                    U[i_s * m + j_s, col] = 1
                    U[(i_s + 1) * m + j_s, col] = -1
                    V[(j_s + 1) * p + (k_s + 1), col] = 1
                    W[k_s * n + i_s, col] = 1
                    W[(k_s + 1) * n + (i_s + 1), col] = 1
                    col += 1
                else:
                    for di in range(is_):
                        for dj in range(js_):
                            for dk in range(ks_):
                                i = i_s + di
                                j = j_s + dj
                                k = k_s + dk
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1

    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    if not check_exact(U, V, W):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
