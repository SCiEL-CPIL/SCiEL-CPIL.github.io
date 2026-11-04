import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p  # 96
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # M(3,2,2) rank-10 decomposition
    # X(3x2) * Y(2x2) = Z(3x2)
    # U1: 6x10, V1: 4x10, W1: 4x10
    U1 = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    ], dtype=np.float64)
    V1 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ], dtype=np.float64)
    W1 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ], dtype=np.float64)

    # Verify M(3,2,2)
    T1 = np.zeros((6, 4, 4), dtype=np.float64)
    for i in range(3):
        for a in range(2):
            for b in range(2):
                T1[i*2+a, a*2+b, b*2+i] = 1.0
    T1_approx = np.einsum('ic,jc,kc->ijk', U1, V1, W1)
    if not np.allclose(T1_approx, T1, atol=1e-10):
        # Try different rank-10 decomposition
        U1 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        ], dtype=np.float64)
        V1 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float64)
        W1 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float64)
        T1_approx = np.einsum('ic,jc,kc->ijk', U1, V1, W1)
        if not np.allclose(T1_approx, T1, atol=1e-10):
            # Use schoolbook for M(3,2,2)
            U1 = np.zeros((6, 12))
            V1 = np.zeros((4, 12))
            W1 = np.zeros((4, 12))
            col = 0
            for i in range(3):
                for a in range(2):
                    for b in range(2):
                        U1[i*2+a, col] = 1
                        V1[a*2+b, col] = 1
                        W1[b*2+i, col] = 1
                        col += 1

    # M(1,2,4) rank-8 decomposition (schoolbook)
    # X(1x2) * Y(2x4) = Z(1x4)
    U2 = np.zeros((2, 8))
    V2 = np.zeros((8, 8))
    W2 = np.zeros((4, 8))
    col = 0
    for a in range(2):
        for b in range(4):
            U2[a, col] = 1
            V2[a*4+b, col] = 1
            W2[b, col] = 1
            col += 1

    # Verify M(1,2,4)
    T2 = np.zeros((2, 8, 4), dtype=np.float64)
    for a in range(2):
        for b in range(4):
            T2[a, a*4+b, b] = 1.0
    T2_approx = np.einsum('ic,jc,kc->ijk', U2, V2, W2)
    if not np.allclose(T2_approx, T2, atol=1e-10):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Kronecker product: M(3,4,8) = M(3,2,2) ⊗ M(1,2,4)
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 * r2

    # U: shape (n*m, total_rank) = (12, total_rank)
    # Index: i in [0,3), a in [0,2), a2 in [0,2)
    # U[i, a, a2, c1, c2] = U1[i*2+a, c1] * U2[a2, c2]
    # Flat index for (i,a,a2): i*4 + a*2 + a2
    U = np.zeros((n*m, total_rank))
    for i in range(3):
        for a in range(2):
            for a2 in range(2):
                row = i * 4 + a * 2 + a2
                for c1 in range(r1):
                    for c2 in range(r2):
                        col = c1 * r2 + c2
                        U[row, col] = U1[i*2+a, c1] * U2[a2, c2]

    # V: shape (m*p, total_rank) = (32, total_rank)
    # Index: a in [0,2), a2 in [0,2), b in [0,4)
    # V[a, a2, b, c1, c2] = V1[a*2+b, c1] * V2[a2*4+b, c2]
    # Flat index for (a,a2,b): a*8 + a2*4 + b
    V = np.zeros((m*p, total_rank))
    for a in range(2):
        for a2 in range(2):
            for b in range(4):
                row = a * 8 + a2 * 4 + b
                for c1 in range(r1):
                    for c2 in range(r2):
                        col = c1 * r2 + c2
                        V[row, col] = V1[a*2+b, c1] * V2[a2*4+b, c2]

    # W: shape (p*n, total_rank) = (24, total_rank)
    # Index: b in [0,4), i in [0,3)
    # W[b, i, c1, c2] = W1[b*2+i, c1] * W2[b, c2]
    # Flat index for (b,i): b*3 + i
    W = np.zeros((p*n, total_rank))
    for b in range(4):
        for i in range(3):
            row = b * 3 + i
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    W[row, col] = W1[b*2+i, c1] * W2[b, c2]

    # Verify
    T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
    if np.allclose(T_approx, T, atol=1e-10):
        best_rank = total_rank
        best_U, best_V, best_W = U, V, W

    # Round to integers/half-integers
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2

    T_approx = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
    if np.allclose(T_approx, T, atol=1e-10):
        best_U, best_V, best_W = U_r, V_r, W_r

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
