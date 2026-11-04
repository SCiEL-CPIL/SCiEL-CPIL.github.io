import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Build the rank-40 decomposition:
    # For each pair of columns (k0, k1) in B:
    #   - Strassen (rank 7) for A[:, 0:2] @ B[0:2, k0:k1+1]
    #   - Rank-1 (rank 1) for A[:, 2] @ B[2, k0:k1+1]
    # Total: 5 * (7 + 1) = 40

    U, V, W, rank = _build_rank40(n, m, p)

    # Validate the decomposition
    if _validate(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback to trivial rank-60 decomposition
    U, V, W, rank = _build_trivial(n, m, p)
    return {"U": U, "V": V, "W": W}


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _build_rank40(n, m, p):
    """
    Build a rank-40 decomposition for 2x3 * 3x10.
    
    A is 2x3, B is 3x10, C is 2x10.
    We split B's columns into 5 pairs. For each pair (k0, k1):
      - Use Strassen (rank 7) for A[:, 0:2] @ B[0:2, k0:k1+1]
      - Use rank-1 (rank 1) for A[:, 2] @ B[2, k0:k1+1]
    Total: 5 * (7 + 1) = 40
    """
    assert n == 2 and m == 3 and p == 10

    rank = 40
    U = [[0] * rank for _ in range(n * m)]   # 6 x 40
    V = [[0] * rank for _ in range(m * p)]   # 30 x 40
    W = [[0] * rank for _ in range(p * n)]   # 20 x 40

    r = 0

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        # U: A00(1), A11(1)  -> rows 0*3+0=0, 1*3+1=4
        # V: B0,k0(1), B1,k1(1) -> rows 0*10+k0=k0, 1*10+k1=10+k1
        # W: C0,k0(1), C1,k1(1) -> rows k0*2+0, k1*2+1
        U[0][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        V[10 + k1][r] += 1
        W[k0 * 2 + 0][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1

        # M2: (A10 + A11) * B0,k0
        # U: A10(1), A11(1) -> rows 1*3+0=3, 1*3+1=4
        # V: B0,k0(1) -> row 0*10+k0=k0
        # W: C1,k0(1), C1,k1(-1) -> rows k0*2+1, k1*2+1
        U[3][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        W[k0 * 2 + 1][r] += 1
        W[k1 * 2 + 1][r] -= 1
        r += 1

        # M3: A00 * (B0,k1 - B1,k1)
        # U: A00(1) -> row 0*3+0=0
        # V: B0,k1(1), B1,k1(-1) -> rows 0*10+k1=k1, 1*10+k1=10+k1
        # W: C0,k1(1), C1,k1(1) -> rows k1*2+0, k1*2+1
        U[0][r] += 1
        V[k1][r] += 1
        V[10 + k1][r] -= 1
        W[k1 * 2 + 0][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1

        # M4: A11 * (B1,k0 - B0,k0)
        # U: A11(1) -> row 1*3+1=4
        # V: B1,k0(1), B0,k0(-1) -> rows 1*10+k0=10+k0, 0*10+k0=k0
        # W: C0,k0(1), C1,k0(1) -> rows k0*2+0, k0*2+1
        U[4][r] += 1
        V[10 + k0][r] += 1
        V[k0][r] -= 1
        W[k0 * 2 + 0][r] += 1
        W[k0 * 2 + 1][r] += 1
        r += 1

        # M5: (A00 + A01) * B1,k1
        # U: A00(1), A01(1) -> rows 0*3+0=0, 0*3+1=1
        # V: B1,k1(1) -> row 1*10+k1=10+k1
        # W: C0,k0(-1), C0,k1(1) -> rows k0*2+0, k1*2+0
        U[0][r] += 1
        U[1][r] += 1
        V[10 + k1][r] += 1
        W[k0 * 2 + 0][r] -= 1
        W[k1 * 2 + 0][r] += 1
        r += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        # U: A10(1), A00(-1) -> rows 1*3+0=3, 0*3+0=0
        # V: B0,k0(1), B0,k1(1) -> rows 0*10+k0=k0, 0*10+k1=k1
        # W: C1,k1(1) -> row k1*2+1
        U[3][r] += 1
        U[0][r] -= 1
        V[k0][r] += 1
        V[k1][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        # U: A01(1), A11(-1) -> rows 0*3+1=1, 1*3+1=4
        # V: B1,k0(1), B1,k1(1) -> rows 1*10+k0=10+k0, 1*10+k1=10+k1
        # W: C0,k0(1) -> row k0*2+0
        U[1][r] += 1
        U[4][r] -= 1
        V[10 + k0][r] += 1
        V[10 + k1][r] += 1
        W[k0 * 2 + 0][r] += 1
        r += 1

        # Rank-1 for A[:,2] * B[2, k0:k1+1]
        # U: A02(1), A12(1) -> rows 0*3+2=2, 1*3+2=5
        # V: B2,k0(1), B2,k1(1) -> rows 2*10+k0=20+k0, 2*10+k1=20+k1
        # W: C0,k0(1), C1,k0(1), C0,k1(1), C1,k1(1) -> rows k0*2+0, k0*2+1, k1*2+0, k1*2+1
        U[2][r] += 1
        U[5][r] += 1
        V[20 + k0][r] += 1
        V[20 + k1][r] += 1
        W[k0 * 2 + 0][r] += 1
        W[k0 * 2 + 1][r] += 1
        W[k1 * 2 + 0][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"

    return U, V, W, rank
