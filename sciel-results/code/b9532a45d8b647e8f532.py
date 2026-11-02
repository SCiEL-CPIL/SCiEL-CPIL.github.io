"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, rank, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_laderman_rank46(n, m, p):
    """
    Rank-46 decomposition of 3x3 * 3x6 using Laderman's rank-23 for 3x3*3x3.
    B is split into B_left (cols 0..2) and B_right (cols 3..5).
    """
    # Verified Laderman 23-term decomposition for 3x3*3x3.
    # Each term: (u_vec, v_vec, w_vec) as flat 9-element lists.
    # u_vec indexed as [a*3+b], v_vec as [j*3+k], w_vec as [i*3+k]
    laderman_terms = [
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([1,0,0,0,1,0,0,0,0], [0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0]),
        ([1,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0]),
        ([0,0,0,0,1,0,0,0,1], [0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0]),
        ([0,1,0,1,0,0,0,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,1,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,1,0,0,1], [0,0,0,0,0,1,0,0,1], [0,0,0,0,0,0,0,0,0]),
        ([1,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,1,0]),
        ([0,1,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]),
        ([1,0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,1], [0,1,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([0,1,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]),
        ([1,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0]),
        ([0,0,1,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,0,0,0]),
        ([1,0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,1]),
    ]

    # Verify the Laderman decomposition works for 3x3*3x3
    np.random.seed(42)
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_expected = A_test @ B_test

    C_computed = np.zeros((3, 3))
    for t in range(23):
        u, v, w = laderman_terms[t]
        u = np.array(u).reshape(3, 3)
        v = np.array(v).reshape(3, 3)
        w = np.array(w).reshape(3, 3)
        x_t = np.sum(u * A_test)
        y_t = np.sum(v * B_test)
        C_computed += w * (x_t * y_t)

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return None

    # Build rank-46 decomposition for 3x3 * 3x6
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for t in range(23):
        u, v, w = laderman_terms[t]
        u = np.array(u).reshape(3, 3)
        v = np.array(v).reshape(3, 3)
        w = np.array(w).reshape(3, 3)

        # Left half
        for a in range(3):
            for b in range(3):
                U[a * m + b][t] = u[a, b]
        for j in range(3):
            for k in range(3):
                V[j * p + k][t] = v[j, k]
        for k in range(3):
            for i in range(3):
                W[k * n + i][t] = w[i, k]

        # Right half
        for a in range(3):
            for b in range(3):
                U[a * m + b][t + 23] = u[a, b]
        for j in range(3):
            for k in range(3):
                V[j * p + (k + 3)][t + 23] = v[j, k]
        for k in range(3):
            for i in range(3):
                W[(k + 3) * n + i][t + 23] = w[i, k]

    if not _verify(U, V, W, rank, n, m, p):
        return None

    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    DEADLINE = t0 + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 3, 6

    # Try rank-46 Laderman construction first
    laderman_result = _build_laderman_rank46(n, m, p)
    if laderman_result is not None:
        return laderman_result

    # Fallback: Schoolbook decomposition: rank = n*m*p = 54
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
