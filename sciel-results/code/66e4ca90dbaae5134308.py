"""
Matrix multiplication 2x4x7 via tensor decomposition.
Builds a rank-50 Strassen-based decomposition and verifies it.
Attempts alternative block decompositions to find lower rank.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def build_strassen_decomp():
        """Build the rank-50 Strassen-based decomposition."""
        rank0 = 50
        U0 = [[0.0] * rank0 for _ in range(n * m)]
        V0 = [[0.0] * rank0 for _ in range(m * p)]
        W0 = [[0.0] * rank0 for _ in range(p * n)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
            U0[i0*m + j0][col] += 1
            U0[(i0+1)*m + (j0+1)][col] += 1
            V0[j0*p + k0][col] += 1
            V0[(j0+1)*p + (k0+1)][col] += 1
            W0[k0*n + i0][col] += 1
            W0[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
            U0[(i0+1)*m + j0][col] += 1
            U0[(i0+1)*m + (j0+1)][col] += 1
            V0[j0*p + k0][col] += 1
            W0[k0*n + (i0+1)][col] += 1
            W0[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
            U0[i0*m + j0][col] += 1
            V0[j0*p + (k0+1)][col] += 1
            V0[(j0+1)*p + (k0+1)][col] -= 1
            W0[(k0+1)*n + i0][col] += 1
            W0[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
            U0[(i0+1)*m + (j0+1)][col] += 1
            V0[(j0+1)*p + k0][col] += 1
            V0[j0*p + k0][col] -= 1
            W0[k0*n + i0][col] += 1
            W0[k0*n + (i0+1)][col] += 1
            col += 1
            # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
            U0[i0*m + j0][col] += 1
            U0[i0*m + (j0+1)][col] += 1
            V0[(j0+1)*p + (k0+1)][col] += 1
            W0[k0*n + i0][col] -= 1
            W0[(k0+1)*n + i0][col] += 1
            col += 1
            # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
            U0[(i0+1)*m + j0][col] += 1
            U0[i0*m + j0][col] -= 1
            V0[j0*p + k0][col] += 1
            V0[j0*p + (k0+1)][col] += 1
            W0[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
            U0[i0*m + (j0+1)][col] += 1
            U0[(i0+1)*m + (j0+1)][col] -= 1
            V0[(j0+1)*p + k0][col] += 1
            V0[(j0+1)*p + (k0+1)][col] += 1
            W0[k0*n + i0][col] += 1
            col += 1

        # 2x4x4: four 2x2x2 Strassen blocks
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        # 2x4x2: two 2x2x2 Strassen blocks
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        # 2x4x1: eight rank-1 terms
        for i in range(n):
            for j in range(m):
                U0[i*m + j][col] = 1
                V0[j*p + 6][col] = 1
                W0[6*n + i][col] = 1
                col += 1

        assert col == rank0, f"Rank mismatch: used {col}, expected {rank0}"
        return U0, V0, W0, rank0

    def verify(U, V, W, n, m, p, rank):
        """Verify that the decomposition gives T[i,j,k]=1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_naive():
        """Build the naive rank-56 decomposition."""
        naive_rank = n * m * p
        U = [[0] * naive_rank for _ in range(n * m)]
        V = [[0] * naive_rank for _ in range(m * p)]
        W = [[0] * naive_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, naive_rank

    # Build the rank-50 Strassen decomposition
    U0, V0, W0, rank0 = build_strassen_decomp()

    # Verify
    if not verify(U0, V0, W0, n, m, p, rank0):
        U, V, W, rank = build_naive()
        return {"U": U, "V": V, "W": W}

    best_rank = rank0
    best_result = {"U": U0, "V": V0, "W": W0}

    # Check deadline
    if time.time() > deadline - 30:
        return best_result

    # Try alternative block decompositions to see if we can get rank < 50
    # Strategy: try different ways to split the 4x7 submatrix into blocks
    # and compute the total rank for each splitting.

    def rank_2_2_2():
        return 7

    def rank_2_2_1():
        return 4

    def rank_2_1_2():
        return 4

    def rank_2_1_1():
        return 2

    def rank_2_m_p(m, p):
        """Compute upper bound on R(2,m,p) via recursive block splitting."""
        if m == 1:
            return p * 2  # naive
        if p == 1:
            return m * 2  # naive
        if m == 2 and p == 2:
            return 7
        if m == 2 and p == 1:
            return 4
        if m == 1 and p == 2:
            return 4
        if m == 1 and p == 1:
            return 2
        # Try all splits
        best = m * p * 2  # naive
        for ms in range(1, m):
            for ps in range(1, p):
                r = rank_2_m_p(ms, ps) + rank_2_m_p(m-ms, p-ps) + rank_2_m_p(ms, p-ps) + rank_2_m_p(m-ms, ps)
                if r < best:
                    best = r
        return best

    # Try a few specific decompositions
    # Decomposition A: 2x4x7 = 2x(2+2)x(4+2+1) -> current, rank 50
    # Decomposition B: 2x4x7 = 2x(3+1)x(4+3)
    rank_B = rank_2_m_p(3, 4) + rank_2_m_p(3, 3) + rank_2_m_p(1, 4) + rank_2_m_p(1, 3)
    
    # Decomposition C: 2x4x7 = 2x(4)x(3+4)
    rank_C = rank_2_m_p(4, 3) + rank_2_m_p(4, 4)
    
    # Decomposition D: 2x4x7 = 2x(2+2)x(3+4)
    rank_D = rank_2_m_p(2, 3) + rank_2_m_p(2, 3) + rank_2_m_p(2, 4) + rank_2_m_p(2, 4)
    
    # Decomposition E: 2x4x7 = 2x(1+3)x(2+5)
    rank_E = rank_2_m_p(1, 2) + rank_2_m_p(1, 5) + rank_2_m_p(3, 2) + rank_2_m_p(3, 5)
    
    # Decomposition F: 2x4x7 = 2x(2+2)x(2+5)
    rank_F = rank_2_m_p(2, 2) + rank_2_m_p(2, 5) + rank_2_m_p(2, 2) + rank_2_m_p(2, 5)
    
    # Decomposition G: 2x4x7 = 2x(4)x(2+2+3)
    rank_G = rank_2_m_p(4, 2) + rank_2_m_p(4, 2) + rank_2_m_p(4, 3)
    
    # Decomposition H: 2x4x7 = 2x(1+1+2)x(2+2+3)
    rank_H = (rank_2_m_p(1,2)+rank_2_m_p(1,2)+rank_2_m_p(1,3) +
              rank_2_m_p(1,2)+rank_2_m_p(1,2)+rank_2_m_p(1,3) +
              rank_2_m_p(2,2)+rank_2_m_p(2,2)+rank_2_m_p(2,3))
    
    # Decomposition I: 2x4x7 = 2x(2+1+1)x(3+3+1)
    rank_I = (rank_2_m_p(2,3)+rank_2_m_p(2,3)+rank_2_m_p(2,1) +
              rank_2_m_p(1,3)+rank_2_m_p(1,3)+rank_2_m_p(1,1) +
              rank_2_m_p(1,3)+rank_2_m_p(1,3)+rank_2_m_p(1,1))
    
    candidates = {
        'B': rank_B,
        'C': rank_C,
        'D': rank_D,
        'E': rank_E,
        'F': rank_F,
        'G': rank_G,
        'H': rank_H,
        'I': rank_I,
    }
    
    min_alt = min(candidates.values())
    
    if min_alt < best_rank:
        # An alternative decomposition has lower theoretical rank.
        # We would need to actually construct it, but for now,
        # the rank-50 is our verified result.
        pass

    return best_result
