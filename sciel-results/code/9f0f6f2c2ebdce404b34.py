"""Strassen-based rank-65 tensor decomposition for 4x4 times 4x5."""
DIMENSIONS = [4, 4, 5]

def _kron(U, V):
    n, m = len(U), len(U[0])
    p, q = len(V), len(V[0])
    out = [[0] * (m * q) for _ in range(n * p)]
    for i in range(n):
        for j in range(m):
            if U[i][j] == 0:
                continue
            for a in range(p):
                for b in range(q):
                    if V[a][b] == 0:
                        continue
                    out[i * p + a][j * q + b] += U[i][j] * V[a][b]
    return out


def construct():
    # 2x2x2 Strassen factor matrices (indices 0..6)
    U2 = [[1, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 1]]
    V2 = [[1, 0, 0, 0, 0, 1, 0],
          [0, 1, -1, -1, 1, 0, 0],
          [0, 0, 1, 0, 0, 0, 1],
          [0, 0, 0, 1, 1, 0, 1]]
    W2 = [[1, 0, 0, -1, 1, 0, -1],
          [0, 0, 1, 1, 0, 0, 1],
          [1, 0, -1, 0, 0, 1, 0],
          [-1, 1, 0, 1, 0, 1, 0]]

    # 4x4x4 via Kronecker product (rank 49)
    U44 = _kron(U2, U2)
    V44 = _kron(V2, V2)
    W44 = _kron(W2, W2)
    r0 = 49

    # Add 4x4x1 schoolbook terms (rank 16) for the 5th column of B
    U = [row[:] + [0] * 16 for row in U44]
    V = [row[:] + [0] * 16 for row in V44]
    W = [row[:] + [0] * 16 for row in W44]
    col = r0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1  # k=4, index k*n+i = 16+i
            col += 1

    return {"U": U, "V": V, "W": W}
