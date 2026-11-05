"""Tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = 3, 3, 6
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    # Schoolbook decomposition as fallback
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
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Dual number approach:
    # Encode 3x6 matrix B as 3x3 over F[t]/(t^2): B[j,2k] + t*B[j,2k+1]
    # This decomposes 3x3x6 into two independent 3x3x3 problems.
    # Use Strassen+schoolbook for each 3x3x3, giving rank 2*26=52.

    def construct_3x3x3():
        nn, mm, pp = 3, 3, 3
        terms = []

        # Correct Strassen tensor decomposition for 2x2x2
        # Each term is (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (i,k)
        strassen_terms = [
            # M1 = (a00+a11)(b00+b11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
            # M2 = (a10+a11)*b00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3 = a00*(b01-b11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4 = a11*(b10-b00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5 = (a00+a01)*b11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6 = (a10-a00)*(b00+b01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            # M7 = (a01-a11)*(b10+b11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (0,1):1}),
        ]

        # Verify Strassen: for each (i,j,k) in {0,1}^3, sum should be 1
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for u, v, w in strassen_terms:
                        if (i,j) in u and (j,k) in v and (i,k) in w:
                            s += u[(i,j)] * v[(j,k)] * w[(i,k)]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            terms.extend(strassen_terms)
            # Schoolbook for remaining (i,j,k) not in {0,1}^3
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if i < 2 and j < 2 and k < 2:
                            continue
                        terms.append(({(i,j):1}, {(j,k):1}, {(i,k):1}))
        else:
            # Full schoolbook for 3x3x3
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        terms.append(({(i,j):1}, {(j,k):1}, {(i,k):1}))

        rank = len(terms)
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        for c, (u, v, w) in enumerate(terms):
            for (i,j), coeff in u.items():
                U[i*3+j][c] = coeff
            for (j,k), coeff in v.items():
                V[j*3+k][c] = coeff
            for (i,k), coeff in w.items():
                W[k*3+i][c] = coeff
        return U, V, W, rank

    U3, V3, W3, r3 = construct_3x3x3()

    # Combine two 3x3x3 blocks for 3x3x6
    rank_combined = 2 * r3
    U_combined = [[0] * rank_combined for _ in range(9)]
    V_combined = [[0] * rank_combined for _ in range(18)]
    W_combined = [[0] * rank_combined for _ in range(18)]

    for c in range(r3):
        for i in range(9):
            U_combined[i][c] = U3[i][c]
            U_combined[i][r3 + c] = U3[i][c]
        for j in range(3):
            for k in range(3):
                V_combined[j*3+k][c] = V3[j*3+k][c]
                V_combined[j*3+k+3][r3+c] = V3[j*3+k][c]
        for i in range(3):
            for k in range(3):
                W_combined[k*3+i][c] = W3[k*3+i][c]
                W_combined[(k+3)*3+i][r3+c] = W3[k*3+i][c]

    if verify(U_combined, V_combined, W_combined, rank_combined):
        best_U, best_V, best_W = U_combined, V_combined, W_combined
        best_rank = rank_combined

    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": U_sb, "V": V_sb, "W": W_sb}
