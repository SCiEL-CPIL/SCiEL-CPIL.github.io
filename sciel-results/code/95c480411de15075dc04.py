import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    
    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2x2.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices for outputs
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Build the known rank-55 solution
    U, V, W = _build_strassen55(n, m, p)
    
    return {"U": U, "V": V, "W": W}
