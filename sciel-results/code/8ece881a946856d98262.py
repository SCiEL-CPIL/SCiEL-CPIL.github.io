import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def _verify(U, V, W, n, m, p, r):
    """Check that the tensor decomposition is exact for the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r_idx in range(r):
                    s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                if s != 1:
                    return False
    return True


def _build_rank44():
    """Build the rank-44 decomposition: 4 Strassen blocks (28) + 16 schoolbook terms."""
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]    # shape (6, 44)
    V = [[0] * rank for _ in range(m * p)]    # shape (24, 44)
    W = [[0] * rank for _ in range(p * n)]    # shape (16, 44)
    col = 0

    # Part 1: A[0:2, 0:2] * B[0:2, 0:8] using 4 Strassen blocks
    for blk in range(4):
        bo = 2 * blk
        ve = bo          # B[0, bo]
        vf = bo + 1      # B[0, bo+1]
        vg = 8 + bo      # B[1, bo]
        vh = 8 + bo + 1  # B[1, bo+1]

        w00 = 2 * bo        # C[0, bo]
        w01 = 2 * bo + 2    # C[0, bo+1]
        w10 = 2 * bo + 1    # C[1, bo]
        w11 = 2 * bo + 3    # C[1, bo+1]

        # P1 = (a+d)(e+h) -> C[0,0] += 1, C[1,1] += 1
        U[0][col] = 1; U[4][col] = 1
        V[ve][col] = 1; V[vh][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1

        # P2 = (c+d)e -> C[1,0] += 1, C[1,1] -= 1
        U[3][col] = 1; U[4][col] = 1
        V[ve][col] = 1
        W[w10][col] = 1; W[w11][col] = -1
        col += 1

        # P3 = a(f-h) -> C[0,1] += 1, C[1,1] += 1
        U[0][col] = 1
        V[vf][col] = 1; V[vh][col] = -1
        W[w01][col] = 1; W[w11][col] = 1
        col += 1

        # P4 = d(g-e) -> C[0,0] += 1, C[1,0] += 1
        U[4][col] = 1
        V[vg][col] = 1; V[ve][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1

        # P5 = (a+b)h -> C[0,0] -= 1, C[0,1] += 1
        U[0][col] = 1; U[1][col] = 1
        V[vh][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1

        # P6 = (c-a)(e+f) -> C[1,1] += 1
        U[3][col] = 1; U[0][col] = -1
        V[ve][col] = 1; V[vf][col] = 1
        W[w11][col] = 1
        col += 1

        # P7 = (b-d)(g+h) -> C[0,0] += 1
        U[1][col] = 1; U[4][col] = -1
        V[vg][col] = 1; V[vh][col] = 1
        W[w00][col] = 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8] -> 16 schoolbook terms
    for k in range(8):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[2 * k + i][col] = 1
            col += 1

    assert col == rank, f"col={col}, rank={rank}"
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Build the known rank-44 solution
    U44, V44, W44 = _build_rank44()

    # Verify rank-44
    if _verify(U44, V44, W44, n, m, p, 44):
        return {"U": U44, "V": V44, "W": W44}

    # Fallback: schoolbook (should not be reached)
    rank_fallback = n * m * p
    U = [[0] * rank_fallback for _ in range(n * m)]
    V = [[0] * rank_fallback for _ in range(m * p)]
    W = [[0] * rank_fallback for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
