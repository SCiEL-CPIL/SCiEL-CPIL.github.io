import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank 56 decomposition using 8 blocks of Strassen 2x2x2
    # This is equivalent to 2 blocks of 2x2x8, each with rank 28
    # R(2,2,8) = 4 * R(2,2,2) = 4 * 7 = 28
    # Total rank = 2 * 28 = 56
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 56
    V = np.zeros((m * p, rank), dtype=float)  # 32 x 56
    W = np.zeros((p * n, rank), dtype=float)  # 16 x 56
    
    # Strassen 2x2x2 pattern
    # U_r[i*2+j], V_r[j*2+k], W_r[k*2+i]
    # Indices for 2x2x2:
    # U: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,2,3
    # V: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,2,3
    # W: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,2,3
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # U entries (coefficients for A):
    # M1: A11, A22
    # M2: A21, A22
    # M3: A11
    # M4: A22
    # M5: A11, A12
    # M6: A21, A11
    # M7: A12, A22
    
    # V entries (coefficients for B):
    # M1: B11, B22
    # M2: B11
    # M3: B21, B22
    # M4: B12, B11
    # M5: B22
    # M6: B11, B12
    # M7: B21, B22
    
    # W entries (coefficients for C):
    # M1: C11, C22
    # M2: C21
    # M3: C12
    # M4: C11, C21
    # M5: C12, C22
    # M6: C11, C22
    # M7: C11, C22
    
    # Let's define the sparse structure
    # u_entries: list of (idx, val)
    # v_entries: list of (idx, val)
    # w_entries: list of (idx, val)
    
    S = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)]),
    ]
    
    # Note: The W indices in Strassen are for C.
    # C11 is at (0,0) -> idx 0
    # C12 is at (0,1) -> idx 1
    # C21 is at (1,0) -> idx 2
    # C22 is at (1,1) -> idx 3
    
    # My W entries above:
    # M1: C11, C22 -> (0,1), (3,1)
    # M2: C21 -> (2,1)
    # M3: C12 -> (1,1)
    # M4: C11, C21 -> (0,1), (2,1) ... Wait, M4 is subtracted in C11 and added in C21.
    # C11 = M1 - M4 - M6 + M7
    # C21 = M2 + M4
    # So M4 contributes +1 to C21 and -1 to C11.
    # So W for M4 should be [(2, 1), (0, -1)].
    # My S[3] W is [(0, -1), (2, 1)]. Correct.
    
    # M5: C12, C22
    # C12 = M3 + M5
    # C22 = M1 - M3 - M5 + M6
    # So M5 contributes +1 to C12 and -1 to C22.
    # W for M5: [(1, 1), (3, -1)].
    # My S[4] W is [(1, 1), (3, -1)]. Correct.
    
    # M6: C11, C22
    # C11 = ... - M6 ...
    # C22 = ... + M6 ...
    # W for M6: [(0, -1), (3, 1)].
    # My S[5] W is [(0, -1), (3, 1)]. Correct.
    
    # M7: C11, C22
    # C11 = ... + M7 ...
    # C22 = ... - M7 ... (Wait, C22 = M1 - M3 - M5 + M6. No M7 in C22?)
    # Let's re-check Strassen.
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # M7 is only in C11.
    # So W for M7: [(0, 1)].
    # My S[6] W is [(0, 1), (3, -1)]. This is WRONG.
    # M7 should only affect C11.
    
    # Let's fix S[6].
    # M7 = (A12-A22)(B21+B22)
    # It contributes to C11 with +1.
    # It does NOT contribute to C22.
    
    # Corrected S:
    S = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),                 # M2
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),                # M3
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (2, 1)]),       # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),        # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),        # M7
    ]

    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            u_entries, v_entries, w_entries = S[r]
            for idx, val in u_entries:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_off, lj + j_off
                U[gi * 4 + gj][cr] += val
            for idx, val in v_entries:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_off, lk + k_off
                V[gj * 8 + gk][cr] += val
            for idx, val in w_entries:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_off, li + i_off
                W[gk * 2 + gi][cr] += val

    # 8 blocks of 2x2x2
    # j_off in {0, 2}, k_off in {0, 2, 4, 6}
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(block_idx * 7, 0, j_off, k_off)
            block_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
