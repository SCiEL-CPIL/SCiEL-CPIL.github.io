"""Deterministic rank-78 decomposition for 3x4*4x7 matrix multiplication.
Uses 6 disjoint 2x2x2 Strassen blocks + 36 schoolbook terms.
"""
import time

N, M, P = 3, 4, 7
RANK = 78
_DEADLINE = 130.0


def construct():
    t0 = time.time()

    # Build factor matrices as dense lists
    U = [[0] * RANK for _ in range(N * M)]
    V = [[0] * RANK for _ in range(M * P)]
    W = [[0] * RANK for _ in range(P * N)]

    col = 0
    covered = set()

    def strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0*M+j0][col] = 1; U[i1*M+j1][col] = 1
        V[j0*P+k0][col] = 1; V[j1*P+k1][col] = 1
        W[k0*N+i0][col] = 1; W[k1*N+i1][col] = 1
        col += 1
        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1*M+j0][col] = 1; U[i1*M+j1][col] = 1
        V[j0*P+k0][col] = 1
        W[k0*N+i1][col] = 1; W[k1*N+i1][col] = -1
        col += 1
        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0*M+j0][col] = 1
        V[j0*P+k1][col] = 1; V[j1*P+k1][col] = -1
        W[k1*N+i0][col] = 1; W[k1*N+i1][col] = 1
        col += 1
        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1*M+j1][col] = 1
        V[j1*P+k0][col] = 1; V[j0*P+k0][col] = -1
        W[k0*N+i0][col] = 1; W[k0*N+i1][col] = 1
        col += 1
        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0*M+j0][col] = 1; U[i0*M+j1][col] = 1
        V[j1*P+k1][col] = 1
        W[k0*N+i0][col] = -1; W[k1*N+i0][col] = 1
        col += 1
        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1*M+j0][col] = 1; U[i0*M+j0][col] = -1
        V[j0*P+k0][col] = 1; V[j0*P+k1][col] = 1
        W[k1*N+i1][col] = 1
        col += 1
        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0*M+j1][col] = 1; U[i1*M+j1][col] = -1
        V[j1*P+k0][col] = 1; V[j1*P+k1][col] = 1
        W[k0*N+i0][col] = 1
        col += 1

        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # 6 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    for blk in blocks:
        strassen_block(*blk)

    # Schoolbook for remaining entries
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) not in covered:
                    U[i*M+j][col] = 1
                    V[j*P+k][col] = 1
                    W[k*N+i][col] = 1
                    col += 1

    assert col == RANK, f"Rank mismatch: {col} != {RANK}"

    # Quick verification on a subset for safety
    if time.time() - t0 < _DEADLINE:
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = 0
                    ui = U[i*M+j]
                    vi = V[j*P+k]
                    wi = W[k*N+i]
                    for c in range(RANK):
                        s += ui[c] * vi[c] * wi[c]
                    if s != 1:
                        raise ValueError(f"Fail at ({i},{j},{k}): {s}")

    return {"U": U, "V": V, "W": W}
