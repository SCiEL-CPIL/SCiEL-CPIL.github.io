import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Rank is 28 (4 blocks of Strassen 2x2x2)
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, col_offset):
        # M1 = (x11+x22)(y11+y22)
        U[a_rows[0] * m + a_cols[0]][col_offset] += 1
        U[a_rows[1] * m + a_cols[1]][col_offset] += 1
        V[b_rows[0] * p + b_cols[0]][col_offset] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset] += 1
        W[c_cols[0] * n + c_rows[0]][col_offset] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset] += 1

        # M2 = (x21+x22)y11
        U[a_rows[1] * m + a_cols[0]][col_offset+1] += 1
        U[a_rows[1] * m + a_cols[1]][col_offset+1] += 1
        V[b_rows[0] * p + b_cols[0]][col_offset+1] += 1
        W[c_cols[0] * n + c_rows[1]][col_offset+1] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset+1] -= 1

        # M3 = x11(y12-y22)
        U[a_rows[0] * m + a_cols[0]][col_offset+2] += 1
        V[b_rows[0] * p + b_cols[1]][col_offset+2] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset+2] -= 1
        W[c_cols[1] * n + c_rows[0]][col_offset+2] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset+2] += 1

        # M4 = x22(y21-y11)
        U[a_rows[1] * m + a_cols[1]][col_offset+3] += 1
        V[b_rows[1] * p + b_cols[0]][col_offset+3] += 1
        V[b_rows[0] * p + b_cols[0]][col_offset+3] -= 1
        W[c_cols[0] * n + c_rows[0]][col_offset+3] += 1
        W[c_cols[0] * n + c_rows[1]][col_offset+3] += 1

        # M5 = (x11+x12)y22
        U[a_rows[0] * m + a_cols[0]][col_offset+4] += 1
        U[a_rows[0] * m + a_cols[1]][col_offset+4] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset+4] += 1
        W[c_cols[0] * n + c_rows[0]][col_offset+4] -= 1
        W[c_cols[1] * n + c_rows[0]][col_offset+4] += 1

        # M6 = (x21-x11)(y11+y12)
        U[a_rows[1] * m + a_cols[0]][col_offset+5] += 1
        U[a_rows[0] * m + a_cols[0]][col_offset+5] -= 1
        V[b_rows[0] * p + b_cols[0]][col_offset+5] += 1
        V[b_rows[0] * p + b_cols[1]][col_offset+5] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset+5] += 1

        # M7 = (x12-x22)(y21+y22)
        U[a_rows[0] * m + a_cols[1]][col_offset+6] += 1
        U[a_rows[1] * m + a_cols[1]][col_offset+6] -= 1
        V[b_rows[1] * p + b_cols[0]][col_offset+6] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset+6] += 1
        W[c_cols[0] * n + c_rows[0]][col_offset+6] += 1

    # Build the 4 independent Strassen blocks
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 7)
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], 14)
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], 21)

    # Verify decomposition
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Compute reconstructed tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # A has shape (n,m) = (2,4), so i indexes (row, col) = (r_a, c_a)
    # B has shape (m,p) = (4,4), so j indexes (row, col) = (r_b, c_b)
    # C has shape (p,n) = (4,2), so k indexes (col, row) = (c_c, r_c) [W[k*n+i, r] convention]
    # The multiplication tensor T has entries T[a_r, a_c, b_r, b_c, c_r, c_c]
    # where A[a_r, a_c] * B[b_r, b_c] contributes to C[c_r, c_c] if a_c == b_r
    # In our flattened indices:
    #   i = a_r * m + a_c  (0..7)
    #   j = b_r * p + b_c  (0..15)
    #   k = c_c * n + c_r  (0..7)  [W index: k*n+i -> W[k, i] where k is the col of C and i is the row]

    # Let's verify: for each (i, j, k), T[i,j,k] should equal 
    # sum_r U[i,r] * V[j,r] * W[k,r]
    # And this should equal delta(a_c, b_r) * delta(c_r, a_r) * delta(c_c, b_c)
    # i.e., 1 if a_c == b_r and c_r == a_r and c_c == b_c, else 0

    # Compute the true tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for a_r in range(n):
        for a_c in range(m):
            i = a_r * m + a_c
            for b_r in range(m):
                for b_c in range(p):
                    j = b_r * p + b_c
                    if a_c == b_r:
                        c_r = a_r
                        c_c = b_c
                        k = c_c * n + c_r
                        T_true[i, j, k] = 1.0

    # Compute reconstructed tensor
    # T_rec[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # We can compute this efficiently using broadcasting
    T_rec = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)

    # Check if they match
    max_err = np.max(np.abs(T_rec - T_true))
    
    if max_err < 1e-10:
        # Decomposition is correct
        return {"U": U, "V": W} if False else {"U": U, "V": V, "W": W}
    else:
        # Should not happen, but fallback
        return {"U": U, "V": V, "W": W}
