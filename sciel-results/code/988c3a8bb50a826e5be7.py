import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    deadline = 120.0
    margin = 15.0

    # Schoolbook: 63
    rank63 = n*m*p
    U63 = np.zeros((n*m, rank63), dtype=object)
    V63 = np.zeros((m*p, rank63), dtype=object)
    W63 = np.zeros((p*n, rank63), dtype=object)
    for k in range(p):
        for i in range(n):
            for j in range(m):
                col = (k * n + i) * m + j
                U63[i*m+j, col] = 1
                V63[j*p+k, col] = 1
                W63[k*n+i, col] = 1

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() > deadline - margin:
        return to_lists(U63, V63, W63)

    # Rank-29 construction: use rank-20 for 3x3x3, rank-20 for 3x3x3, rank-9 for 3x3x1
    # This requires rank 29
    # For now, only schoolbook is fully implemented.
    # We need to construct a rank-29 if we want to beat schoolbook.
    # The tensor is 3x3x7. The minimal rank is unknown (likely around 44 or 52).
    # But we don't have an explicit rank-29 or 44 decomposition.

    # We should just return schoolbook.

    return to_lists(U63, V63, W63)
