"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check if (U, V, W) is a valid decomposition."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T = np.zeros((n * m, m * p, p * n))
    for t in range(rank):
        for a in range(n * m):
            if U[a][t] != 0:
                for b in range(m * p):
                    if V[b][t] != 0:
                        for c in range(p * n):
                            if W[c][t] != 0:
                                T[a, b, c] += U[a][t] * V[b][t] * W[c][t]
    # Check against the target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i * m + j, j * p + k, k * n + i] != 1:
                    return False
    # Check no other entries are non-zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                for a2 in range(n * m):
                    for b2 in range(m * p):
                        for c2 in range(p * n):
                            if (a2, b2, c2) != (a, b, c):
                                if T[a2, b2, c2] != 0:
                                    return False
    return True

def _try_structured_rank(n, m, p, max_rank, deadline):
    """Try to find a structured decomposition with rank <= max_rank."""
    start = time.time()
    best = None
    # Try ranks from max_rank down to n*m*p
    for rank in range(max_rank, n * m * p + 1):
        if time.time() > deadline:
            break
        # Try random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            # Initialize with random small values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            # Simple gradient-free search: try to reduce rank by zeroing out columns
            # This is a placeholder for a more sophisticated search
            if time.time() > deadline:
                break
    return best

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x8 matrix."""
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=3, p=8

    # Schoolbook decomposition: rank = n*m*p = 72
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

    result = {"U": U, "V": V, "W": W}

    # Check deadline
    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
