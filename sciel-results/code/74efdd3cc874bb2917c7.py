import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 3, 4, 6
    nm, mp, pn = n * m, m * p, p * n

    # Rank 66 block decomposition baseline
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    U = np.zeros((nm, 66), dtype=float)
    V = np.zeros((mp, 66), dtype=float)
    W = np.zeros((pn, 66), dtype=float)
    col = 0

    # Local search: try to reduce rank by dropping redundant columns
    def verify(U, V, W):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.multiply.outer(np.multiply.outer(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :]), np.ones(r))
                    expected = 1.0
                    max_err = max(max_err, abs(s - expected))
        return max_err

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # Strassen 7 terms
        U[A00, col] += 1; U[A11, col] += 1
        V[B00, col] += 1; V[B11, col] += 1
        W[C00, col] += 1; W[C11, col] += 1
        col += 1
        U[A10, col] += 1; U[A11, col] += 1
        V[B00, col] += 1
        W[C10, col] += 1; W[C11, col] -= 1
        col += 1
        U[A00, col] += 1
        V[B01, col] += 1; V[B11, col] -= 1
        W[C01, col] += 1; W[C11, col] += 1
        col += 1
        U[A11, col] += 1
        V[B10, col] += 1; V[B00, col] -= 1
        W[C00, col] += 1; W[C10, col] += 1
        col += 1
        U[A00, col] += 1; U[A01, col] += 1
        V[B11, col] += 1
        W[C00, col] -= 1; W[C01, col] += 1
        col += 1
        U[A10, col] += 1; U[A00, col] -= 1
        V[B00, col] += 1; V[B01, col] += 1
        W[C11, col] += 1
        col += 1
        U[A01, col] += 1; U[A11, col] -= 1
        V[B10, col] += 1; V[B11, col] += 1
        W[C00, col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
        col += 1
        U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
        col += 1
        U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
        col += 1
        U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
        col += 1

    # Row 0 of A (1x4)
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)

    # Rows 1-2 of A (2x4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)

    # Try to reduce rank by local search (drop columns if possible)
    def try_reduce_rank(U, V, W, max_iters=500):
        r = U.shape[1]
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        best_rank = r
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()

        for it in range(max_iters):
            if time.time() > deadline:
                break

            found = False
            for c in range(U_curr.shape[1]):
                if time.time() > deadline:
                    break

                # Try dropping column c
                U_test = np.delete(U_curr, c, axis=1)
                V_test = np.delete(V_curr, c, axis=1)
                W_test = np.delete(W_curr, c, axis=1)

                err = verify(U_test, V_test, W_test)
                if err < 1e-10:
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    best_rank = U_curr.shape[1]
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    found = True
                    break

            if not found:
                break

        return best_rank, best_U, best_V, best_W

    # Attempt rank reduction
    reduced_rank, U_red, V_red, W_red = try_reduce_rank(U, V, W)

    # Verify the reduced result
    if reduced_rank < 66:
        err = verify(U_red, V_red, W_red)
        if err < 1e-10:
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    # Fallback to original rank-66
    err = verify(U, V, W)
    if err > 1e-10:
        # Schoolbook fallback
        rank_fb = n * m * p
        U = np.zeros((nm, rank_fb))
        V = np.zeros((mp, rank_fb))
        W = np.zeros((pn, rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
