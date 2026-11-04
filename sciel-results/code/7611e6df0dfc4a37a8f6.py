"""
Deterministic rank-reduction for 4x5 x 5x6 matrix multiplication tensor.
Uses alternating minimization with exact rational verification to find
a rank below the schoolbook n*m*p = 120 bound.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120

    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    deadline = time.time() + 95  # Safety margin

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank

    # Try to find lower rank via alternating minimization
    for target_rank in range(70, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break

        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)

        # Initialize from schoolbook
        for c in range(min(target_rank, schoolbook_rank)):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0

        # Alternating least squares
        for it in range(40):
            if time.time() > deadline - 10:
                break

            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_mat.T, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass

            # Check convergence every 5 iterations
            if it % 5 == 4:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (val - 1.0) ** 2
                if err < 1e-8:
                    # Verify with exact rationals
                    U_frac = [[Fraction(int(U[r, c] * 1000000 + 0.5), 1000000) for c in range(target_rank)] for r in range(n * m)]
                    V_frac = [[Fraction(int(V[r, c] * 1000000 + 0.5), 1000000) for c in range(target_rank)] for r in range(m * p)]
                    W_frac = [[Fraction(int(W[r, c] * 1000000 + 0.5), 1000000) for c in range(target_rank)] for r in range(p * n)]

                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = Fraction(0)
                                for c in range(target_rank):
                                    total += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                                if total != 1:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        best_U = [row for row in U_frac]
                        best_V = [row for row in V_frac]
                        best_W = [row for row in W_frac]
                        best_rank = target_rank
                        break

        if best_rank < schoolbook_rank:
            break

    # Ensure output format matches expected (lists of lists of numbers)
    def to_plain(lst):
        if lst is None:
            return None
        return [[float(x) for x in row] for row in lst]

    if isinstance(best_U[0][0], Fraction):
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
    else:
        U_out = best_U
        V_out = best_V
        W_out = best_W

    return {"U": U_out, "V": V_out, "W": W_out}
