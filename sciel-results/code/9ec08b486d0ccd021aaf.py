import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25

    # Partition: [2, 2, 1] for each of the three dimensions
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]

    n_offsets = [0, 2, 4]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]

    # Strassen 2x2x2 rank-1 terms
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: dict mapping (di, dj) -> coeff  [for A block]
    # v_entries: dict mapping (dj, dk) -> coeff  [for B block]
    # w_entries: dict mapping (dk, di) -> coeff  [for C block]
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        (
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        ),
        # M2 = (A10+A11)*B00
        (
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 0): 1},
        ),
        # M3 = A00*(B01-B11)
        (
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(0, 1): 1, (1, 1): -1},
        ),
        # M4 = A11*(B10-B00)
        (
            {(1, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(1, 0): 1, (0, 0): -1},
        ),
        # M5 = (A00+A01)*B11
        (
            {(0, 0): 1, (0, 1): 1},
            {(1, 1): 1},
            {(1, 1): 1},
        ),
        # M6 = (A10-A00)*(B00+B01)
        (
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (0, 1): 1},
            {(0, 0): 1, (0, 1): 1},
        ),
        # M7 = (A01-A11)*(B10+B11)
        (
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): 1},
        ),
    ]

    # First pass: count total rank needed
    total_rank = 0
    for ni in range(3):
        for mj in range(3):
            for pk in range(3):
                ns = n_parts[ni]
                ms = m_parts[mj]
                ps = p_parts[pk]
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7
                else:
                    total_rank += ns * ms * ps

    # Allocate matrices
    U = [[0.0] * total_rank for _ in range(N)]
    V = [[0.0] * total_rank for _ in range(M)]
    W = [[0.0] * total_rank for _ in range(P)]

    col = 0

    for ni in range(3):
        for mj in range(3):
            for pk in range(3):
                ns = n_parts[ni]
                ms = m_parts[mj]
                ps = p_parts[pk]

                noff = n_offsets[ni]
                moff = m_offsets[mj]
                poff = p_offsets[pk]

                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen: 7 terms
                    for term in strassen_terms:
                        u_entries, v_entries, w_entries = term
                        for (di, dj), uc in u_entries.items():
                            ui = (noff + di) * m + (moff + dj)
                            U[ui][col] += uc
                        for (dj, dk), vc in v_entries.items():
                            vi = (moff + dj) * p + (poff + dk)
                            V[vi][col] += vc
                        for (dk, di), wc in w_entries.items():
                            wi = (poff + dk) * n + (noff + di)
                            W[wi][col] += wc
                        col += 1
                else:
                    # Schoolbook for this block
                    for di in range(ns):
                        for dj in range(ms):
                            for dk in range(ps):
                                ui = (noff + di) * m + (moff + dj)
                                vi = (moff + dj) * p + (poff + dk)
                                wi = (poff + dk) * n + (noff + di)
                                U[ui][col] = 1
                                V[vi][col] = 1
                                W[wi][col] = 1
                                col += 1

    # Verify the decomposition numerically
    # Tensor check: T[i,k,j] = sum_r U_r[i,k] * V_r[k,j] * W_r[j,i]
    # Must satisfy: sum_{i,k,j} A[i,k] * T[i,k,j] * B[k,j] = sum_{i,j} (A@B)[i,j]
    rng = np.random.default_rng(42)
    A = rng.integers(-3, 4, size=(n, m)).astype(float)
    B = rng.integers(-3, 4, size=(m, p)).astype(float)
    C_expected = A @ B

    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)

    T = np.zeros((n, m, p))
    for r in range(total_rank):
        U_r = U_arr[:, r].reshape(n, m)
        V_r = V_arr[:, r].reshape(m, p)
        W_r = W_arr[:, r].reshape(p, n)
        T += np.einsum('ik,kj,ji->ikj', U_r, V_r, W_r)

    lhs = np.einsum('ik,ikj,kj->', A, T, B)
    rhs = np.sum(C_expected)

    if abs(lhs - rhs) > 1e-6:
        # Fallback to schoolbook
        rank_fallback = n * m * p
        U = [[0.0] * rank_fallback for _ in range(N)]
        V = [[0.0] * rank_fallback for _ in range(M)]
        W = [[0.0] * rank_fallback for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to clean integers
    U_out = [[int(round(x)) for x in row] for row in U]
    V_out = [[int(round(x)) for x in row] for row in V]
    W_out = [[int(round(x)) for x in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
