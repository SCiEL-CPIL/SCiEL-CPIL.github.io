"""Deterministic algebraic construction for 4x4 by 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s kill

    # Build the list of (u, v, w) index triples via direct enumeration
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                triples.append((u, v, w))

    rank = len(triples)

    # Allocate factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Populate: each triple corresponds to one rank column with a single 1 in each factor
    for col, (u, v, w) in enumerate(triples):
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1

    result = {"U": U, "V": V, "W": W}

    # Exhaustive verification if time permits
    if time.time() < deadline - 5:
        target_set = set(triples)
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Check all target entries equal 1
        for (u, v, w) in triples:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Target entry ({u},{v},{w}) = {val}, expected 1")

        # Check all non-target entries equal 0 (exhaustive)
        total_non_target = n * m * m * p * p * n - rank
        count_checked = 0
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    if (u, v, w) in target_set:
                        continue
                    val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
                    if abs(val) > 1e-9:
                        raise ValueError(f"Non-target entry ({u},{v},{w}) = {val}, expected 0")
                    count_checked += 1

    return result
# EVOLVE-BLOCK-END
