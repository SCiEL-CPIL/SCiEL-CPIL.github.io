"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    t0 = time.time()
    DEADLINE = t0 + 140.0

    def elapsed():
        return time.time() - t0

    def schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 48."""
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
        return U, V, W

    def verify_decomp(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def strassen_rank44():
        """
        Construct rank-44 decomposition using block Strassen.
        Split m=3 into m1=1 and m2=2.
        Part 1: 2x1x8 (Rank 16)
        Part 2: 2x2x8, split into 4 blocks of 2x2x2 (Rank 7 each) -> Total 28.
        Total Rank: 44.
        """
        rank = 44
        U = [[0] * rank for _ in range(n * m)]  # 6 x 44
        V = [[0] * rank for _ in range(m * p)]  # 24 x 44
        W = [[0] * rank for _ in range(p * n)]  # 16 x 44

        col = 0

        # --- Part 1: 2x1x8 (j=0) ---
        # A is 2x1, B is 1x8.
        # C[i, k] = A[i, 0] * B[0, k]
        # Rank 16.
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
        if elapsed() > DEADLINE:
            return None

        # --- Part 2: 2x2x8 (j=1,2) ---
        # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
        # Split B into 4 blocks of 2 columns.
        for block_idx in range(4):
            k_start = block_idx * 2
            
            # Local indices mapping:
            # A00: u=1, A01: u=2, A10: u=4, A11: u=5
            # B00: v=8+k_start, B01: v=9+k_start, B10: v=16+k_start, B11: v=17+k_start
            # C00: w=2*k_start, C01: w=2*k_start+1, C10: w=2*k_start+2, C11: w=2*k_start+3

            # Strassen 2x2x2 steps
            # 1. M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
            U[1][col] += 1
            U[5][col] += 1
            V[8 + k_start][col] += 1
            V[17 + k_start][col] += 1
            W[2 * k_start][col] += 1
            W[2 * k_start + 3][col] += 1
            col += 1

            # 2. M2 = (A10+A11)*B00 -> C10+=M2, C11-=M2
            U[4][col] += 1
            U[5][col] += 1
            V[8 + k_start][col] += 1
            W[2 * k_start + 2][col] += 1
            W[2 * k_start + 3][col] -= 1
            col += 1

            # 3. M3 = A00*(B01-B11) -> C01+=M3, C11+=M3
            U[1][col] += 1
            V[9 + k_start][col] += 1
            V[17 + k_start][col] -= 1
            W[2 * k_start + 1][col] += 1
            W[2 * k_start + 3][col] += 1
            col += 1

            # 4. M4 = A11*(B10-B00) -> C00+=M4, C10+=M4
            U[5][col] += 1
            V[16 + k_start][col] += 1
            V[8 + k_start][col] -= 1
            W[2 * k_start][col] += 1
            W[2 * k_start + 2][col] += 1
            col += 1

            # 5. M5 = (A00+A01)*B11 -> C00-=M5, C01+=M5
            U[1][col] += 1
            U[2][col] += 1
            V[17 + k_start][col] += 1
            W[2 * k_start][col] -= 1
            W[2 * k_start + 1][col] += 1
            col += 1

            # 6. M6 = (A10-A00)*(B00+B01) -> C11+=M6
            U[4][col] += 1
            U[1][col] -= 1
            V[8 + k_start][col] += 1
            V[9 + k_start][col] += 1
            W[2 * k_start + 3][col] += 1
            col += 1

            # 7. M7 = (A01-A11)*(B10+B11) -> C00+=M7
            U[2][col] += 1
            U[5][col] -= 1
            V[16 + k_start][col] += 1
            V[17 + k_start][col] += 1
            W[2 * k_start][col] += 1
            col += 1

        if col != rank:
            return None

        return U, V, W

    # 1. Try the deterministic rank-44 construction
    result = strassen_rank44()
    if result is not None:
        U, V, W = result
        if verify_decomp(U, V, W, 44):
            return {"U": U, "V": V, "W": W}

    # 2. Fallback to schoolbook rank 48
    U_sch, V_sch, W_sch = schoolbook()
    return {"U": U_sch, "V": V_sch, "W": W_sch}
# EVOLVE-BLOCK-END
