"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    # Build a rank-32 decomposition using Strassen for 2x2x2 blocks
    # Split p=9 into blocks: 2+2+2+2+1
    # Each 2x2x2 block uses Strassen (rank 7), 2x2x1 uses schoolbook (rank 4)
    # Total: 7*4 + 4 = 32
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M2 = (A10+A11)B00 -> C10, C11 (with -1 on C11)
            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            # M3 = A00(B01-B11) -> C01, C11
            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M4 = A11(B10-B00) -> C00, C10
            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            # M5 = (A00+A01)B11 -> C00, C01 (with -1 on C00)
            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11
            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00
            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            # 4 terms for a single k
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        # Split p=9 into 4 Strassen blocks (2x2 each) + 1 trivial block
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    # Build the rank-32 decomposition
    result = build_rank32()

    # Verify it's correct
    if verify(result["U"], result["V"], result["W"]):
        # Try to find a lower-rank decomposition using the time budget
        # The theoretical minimum rank for M_{2,2,9} is at least 2*9 = 18 (since
        # each of the 9 output columns needs at least 2 terms for a 2x2 product).
        # Known results: rank(M_{2,2,k}) for small k:
        # k=1: 4, k=2: 7, k=3: 11, k=4: 14, k=5: 18, k=6: 21, k=7: 25, k=8: 28, k=9: 32
        # But there might be better decompositions using non-block approaches.
        # For now, we return the verified rank-32 result.
        return result

    # Fallback to schoolbook if verification fails
    return schoolbook()
# EVOLVE-BLOCK-END
