"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    # Check orthogonality: for non-matching indices, sum should be 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            for kp2 in range(p):
                                if i == ip and j == jp and k == kp:
                                    continue
                                s = 0
                                for r in range(rank):
                                    s += U[i * m + j][r] * V[jp * p + k][r] * W[kp2 * n + ip][r]
                                if abs(s) > 1e-9:
                                    return False
    return True

def construct():
    deadline = time.time() + 110  # 110s deadline, 10s safety margin
    n, m, p = DIMENSIONS
    # Return schoolbook decomposition (rank 48)
    result = _schoolbook(n, m, p)
    return result
# EVOLVE-BLOCK-END
