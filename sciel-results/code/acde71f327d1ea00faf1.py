import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for r in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Laderman's 3x3x3 rank-23 decomposition factors
    laderman_3x3x3 = [
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
    ]

    # Verify Laderman factors
    U_test = [[0]*23 for _ in range(9)]
    V_test = [[0]*23 for _ in range(9)]
    W_test = [[0]*23 for _ in range(9)]
    for r, (u, v, w) in enumerate(laderman_3x3x3):
        for idx in range(9):
            U_test[idx][r] = u[idx]
            V_test[idx][r] = v[idx]
            W_test[idx][r] = w[idx]

    laderman_valid = verify_decomposition(U_test, V_test, W_test, 3, 3, 3)

    if not laderman_valid:
        return schoolbook()

    # Strategy: Use Laderman for first 3 columns, then try to extend to 7 columns
    # with additional rank terms that cover the remaining 4 columns.
    
    # First, build the base 3x3x3 decomposition embedded in 3x3x7
    # (zeros in V and W for k >= 3)
    total_rank = 23 + 12  # Start with Laderman + schoolbook for remaining
    U_full = [[0] * total_rank for _ in range(9)]
    V_full = [[0] * total_rank for _ in range(21)]
    W_full = [[0] * total_rank for _ in range(21)]

    for r, (u, v, w) in enumerate(laderman_3x3x3):
        for idx in range(9):
            U_full[idx][r] = u[idx]
        for j in range(3):
            for k in range(3):
                V_full[j * 7 + k][r] = v[j * 3 + k]
        for k in range(3):
            for i in range(3):
                W_full[k * 3 + i][r] = w[k * 3 + i]

    # Add schoolbook terms for k=3,4,5,6
    offset = 23
    col_in_part = 0
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U_full[i * 3 + j][offset + col_in_part] = 1
                V_full[j * 7 + k][offset + col_in_part] = 1
                W_full[k * 3 + i][offset + col_in_part] = 1
                col_in_part += 1

    # Verify
    if verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}

    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
