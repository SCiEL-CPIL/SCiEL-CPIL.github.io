"""Program using a structured row-sum reduction + exact linear solve for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin under the 150 s wall-clock kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42

    def _elapsed():
        return time.time() - start_time

    def _time_left():
        return DEADLINE - _elapsed()

    def _schoolbook_decomposition():
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

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor (trace convention)."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(r):
                        total += Fraction(U[i * m + j][c]) * \
                                   Fraction(V[j * p + k][c]) * \
                                   Fraction(W[k * n + i][c])
                    if total != 1:
                        return False
        return True

    def _gaussian_elimination(A, b):
        """
        Solve A*x = b over rationals. Returns x (list of Fractions) or None if
        inconsistent. Free variables are set to 0.
        """
        num_rows = len(A)
        num_cols = len(A[0])
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        pivot_cols = []
        rank = 0
        for col in range(num_cols):
            pivot_row = None
            for row in range(rank, num_rows):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            pivot_val = aug[rank][col]
            for j in range(col, num_cols + 1):
                aug[rank][j] = aug[rank][j] / pivot_val
            for row in range(num_rows):
                if row != rank and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(col, num_cols + 1):
                        aug[row][j] = aug[row][j] - factor * aug[rank][j]
            pivot_cols.append(col)
            rank += 1

        # Consistency check
        for row in range(rank, num_rows):
            if any(aug[row][j] != 0 for j in range(num_cols)):
                return None

        x = [Fraction(0) for _ in range(num_cols)]
        for ridx, pcol in enumerate(pivot_cols):
            x[pcol] = aug[ridx][num_cols]
        return x

    def solve_for_W(U, V, target_rank):
        """
        Given U (nm x r) and V (mp x r), solve for W (pn x r) such that for all
        (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1.
        Linear system in the entries of W.
        """
        r = target_rank
        num_constraints = n * m * p
        A = [[Fraction(0) for _ in range(p * n * r)] for _ in range(num_constraints)]
        b = [Fraction(1) for _ in range(num_constraints)]

        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        coeff = Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
                        if coeff != 0:
                            pos = (k * n + i) * r + c
                            A[idx][pos] = coeff
                    idx += 1

        w_vec = _gaussian_elimination(A, b)
        if w_vec is None:
            return None

        W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
        for row in range(p * n):
            for col in range(r):
                W[row][col] = w_vec[row * r + col]
        return W

    def try_candidate(U, V):
        """Solve for W for the given U,V and verify. Returns (U,V,W) or None."""
        r = len(U[0])
        W = solve_for_W(U, V, r)
        if W is None:
            return None
        if verify_decomposition(U, V, W):
            return (U, V, W)
        return None

    def build_structured_candidates():
        """
        Row-sum reduction for the 3x7 right factor.
        Compute R1 = A_row1 * B and R2 = A_row2 * B (14 bilinear terms each),
        then form R1-R2, R1+R2, and R3 = R1+R2-(R1-R2). The third row reuses
        the R1,R2 terms, so the 3x7 block needs only 14 terms (not 21).
        Combined with the 2x3 left factor (6 terms) -> total rank 20.
        """
        cands = []

        # ---- Scheme A: R3 = R1 + R2 - (R1 - R2)  (rank 20) ----
        rA = 20
        U = [[0] * rA for _ in range(n * m)]
        V = [[0] * rA for _ in range(m * p)]

        # Left factor 2x3: term (i, j) -> U[i*m+j, t] = 1
        t = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][t] = 1
                t += 1
        # t == 6

        # R1 terms (14): V[j*p+k, t] = 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = 1
                t += 1
        # t == 20

        # R2 terms (14): V[j*p+k, t] = 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = 1
                t += 1
        # t == 34

        # R1-R2 terms (14): V[j*p+k, t] = -1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = -1
                t += 1
        # t == 48

        # R1+R2 terms (14): V[j*p+k, t] = 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = 1
                t += 1
        # t == 62

        cands.append((U, V))

        # ---- Scheme B: R3 = R1 - (R1 - R2)  (rank 18) ----
        rB = 18
        U = [[0] * rB for _ in range(n * m)]
        V = [[0] * rB for _ in range(m * p)]
        t = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][t] = 1
                t += 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = 1
                t += 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = -1
                t += 1
        cands.append((U, V))

        # ---- Scheme C: R3 = R2 - (R2 - R1)  (rank 18) ----
        rC = 18
        U = [[0] * rC for _ in range(n * m)]
        V = [[0] * rC for _ in range(m * p)]
        t = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][t] = 1
                t += 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = 1
                t += 1
        for j in range(3):
            for k in range(7):
                V[j * p + k][t] = -1
                t += 1
        cands.append((U, V))

        return cands

    def build_random_candidates(num):
        """Random rank-41 candidates (schoolbook prefix + random tail) as a fallback."""
        cands = []
        r = schoolbook_rank - 1
        for _ in range(num):
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= r:
                            break
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        col += 1
            for row in range(n * m):
                for c in range(col, r):
                    U[row][c] = Fraction(random.randint(-3, 3), random.randint(1, 3))
            for row in range(m * p):
                for c in range(col, r):
                    V[row][c] = Fraction(random.randint(-3, 3), random.randint(1, 3))
            cands.append((U, V))
        return cands

    # Fallback result (always valid)
    best = _schoolbook_decomposition()

    # Try structured row-sum candidates first (rank 20 / 18)
    for (U, V) in build_structured_candidates():
        if _time_left() < 8.0:
            break
        res = try_candidate(U, V)
        if res is not None:
            U, V, W = res
            rank = len(U[0])
            if rank < len(best["U"][0]):
                best = {"U": U, "V": V, "W": W}

    # If structured succeeded, we are at rank <= 20; stop (no need to search 41).
    if len(best["U"][0]) <= 20:
        return best

    # Otherwise fall back to randomized rank-41 search (bounded by deadline).
    for (U, V) in build_random_candidates(40):
        if _time_left() < 8.0:
            break
        res = try_candidate(U, V)
        if res is not None:
            U, V, W = res
            rank = len(U[0])
            if rank < len(best["U"][0]):
                best = {"U": U, "V": V, "W": W}

    return best
# EVOLVE-BLOCK-END
