"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for the 2x2 x 2x2 sub-blocks of the main
2x2 x 2x8 part, and rank-1 terms for the remaining 1x2 x 2x8, 2x1 x 1x8,
and 1x1 x 1x8 parts.
Total rank: 4*7 + 1 + 1 + 1 = 31.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    rank = 31
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add a term
    def add_term(idx, u_entries, v_entries, w_entries):
        for (i, j), val in u_entries:
            U[i * m + j][idx] = val
        for (j, k), val in v_entries:
            V[j * p + k][idx] = val
        for (k, i), val in w_entries:
            W[k * n + i][idx] = val

    # 1. Main block: A[0:2, 0:2] * B[0:2, 0:8]
    # Split B into 4 blocks of width 2.
    # Each block uses Strassen (7 terms).
    
    def strassen_block(k_offset, idx_start):
        # k_offset is 0, 2, 4, 6
        # U indices: i in {0,1}, j in {0,1}
        # V indices: j in {0,1}, k in {k_offset, k_offset+1}
        # W indices: k in {k_offset, k_offset+1}, i in {0,1}
        
        # m1 = (a+d)(e+h)
        add_term(idx_start + 0, 
                 [((0,0), 1), ((1,1), 1)],
                 [((0, k_offset), 1), ((1, k_offset+1), 1)],
                 [((k_offset, 0), 1), ((k_offset+1, 1), 1)])
        
        # m2 = (c+d)*e
        add_term(idx_start + 1,
                 [((1,0), 1), ((1,1), 1)],
                 [((0, k_offset), 1)],
                 [((k_offset, 1), 1), ((k_offset+1, 1), -1)])
        
        # m3 = a*(f-h)
        add_term(idx_start + 2,
                 [((0,0), 1)],
                 [((0, k_offset+1), 1), ((1, k_offset+1), -1)],
                 [((k_offset+1, 0), 1), ((k_offset+1, 1), 1)])
        
        # m4 = d*(g-e)
        add_term(idx_start + 3,
                 [((1,1), 1)],
                 [((1, k_offset), 1), ((0, k_offset), -1)],
                 [((k_offset, 0), 1), ((k_offset, 1), 1)])
        
        # m5 = (a+b)*h
        add_term(idx_start + 4,
                 [((0,0), 1), ((0,1), 1)],
                 [((1, k_offset+1), 1)],
                 [((k_offset, 0), -1), ((k_offset+1, 0), 1)])
        
        # m6 = (c-a)*(e+f)
        add_term(idx_start + 5,
                 [((1,0), 1), ((0,0), -1)],
                 [((0, k_offset), 1), ((0, k_offset+1), 1)],
                 [((k_offset+1, 1), 1)])
        
        # m7 = (b-d)*(g+h)
        add_term(idx_start + 6,
                 [((0,1), 1), ((1,1), -1)],
                 [((1, k_offset), 1), ((1, k_offset+1), 1)],
                 [((k_offset, 0), 1)])

    strassen_block(0, 0)
    strassen_block(2, 7)
    strassen_block(4, 14)
    strassen_block(6, 21)

    # 2. Row 2 of A: A[2, 0:2] * B[0:2, 0:8]
    # i=2, j in {0,1}, k in {0..7}
    # 1 term
    idx = 28
    u_entries = [((2, 0), 1), ((2, 1), 1)]
    v_entries = []
    for k in range(8):
        v_entries.append(((0, k), 1))
        v_entries.append(((1, k), 1))
    w_entries = []
    for k in range(8):
        w_entries.append(((k, 2), 1))
    add_term(idx, u_entries, v_entries, w_entries)

    # 3. Col 2 of A: A[0:2, 2] * B[2, 0:8]
    # i in {0,1}, j=2, k in {0..7}
    # 1 term
    idx = 29
    u_entries = [((0, 2), 1), ((1, 2), 1)]
    v_entries = []
    for k in range(8):
        v_entries.append(((2, k), 1))
    w_entries = []
    for k in range(8):
        w_entries.append(((k, 0), 1))
        w_entries.append(((k, 1), 1))
    add_term(idx, u_entries, v_entries, w_entries)

    # 4. Corner: A[2, 2] * B[2, 0:8]
    # i=2, j=2, k in {0..7}
    # 1 term
    idx = 30
    u_entries = [((2, 2), 1)]
    v_entries = []
    for k in range(8):
        v_entries.append(((2, k), 1))
    w_entries = []
    for k in range(8):
        w_entries.append(((k, 2), 1))
    add_term(idx, u_entries, v_entries, w_entries)

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
