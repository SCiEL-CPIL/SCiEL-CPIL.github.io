"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy: split B (4x9) into two 4x4 blocks and one 4x1 block.
For each 4x4 block, use recursive Strassen (rank 49).
For the 4x1 block, use schoolbook (rank 16).
Total: 49 + 49 + 16 = 114 rank.
"""

import time


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses recursive Strassen for 4x4*4x4 blocks and schoolbook for 4x4*4x1 block.
    Total rank: 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = 4, 4, 9
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (36, rank)
    # W shape: (p*n, rank) = (36, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

    U_cols = []  # Each element is a list of length 16
    V_cols = []  # Each element is a list of length 36
    W_cols = []  # Each element is a list of length 36

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # 7 Strassen terms for 2x2 * 2x2 -> 2x2
    # Each: (A_mult, B_mult, C_mult) where each is dict {(row,col): coeff}
    # The product A_mult * B_mult contributes to C via C_mult
    strassen_2x2 = [
        # s1 = (a11+a22)(b11+b22): c11 += s1, c22 += s1
        (({(0,0): 1, (1,1): 1},
          {(0,0): 1, (1,1): 1},
          {(0,0): 1, (1,1): 1})),
        # s2 = (a21+a22)b11: c21 += s2, c22 -= s2
        (({(1,0): 1, (1,1): 1},
          {(0,0): 1},
          {(1,0): 1, (1,1): -1})),
        # s3 = a11(b12-b22): c12 += s3, c22 += s3
        (({(0,0): 1},
          {(0,1): 1, (1,1): -1},
          {(0,1): 1, (1,1): 1})),
        # s4 = a22(b21-b11): c11 += s4, c21 += s4
        (({(1,1): 1},
          {(1,0): 1, (0,0): -1},
          {(0,0): 1, (1,0): 1})),
        # s5 = (a11+a12)b22: c11 -= s5, c12 += s5
        (({(0,0): 1, (0,1): 1},
          {(1,1): 1},
          {(0,0): -1, (0,1): 1})),
        # s6 = (a21-a11)(b11+b12): c22 += s6
        (({(1,0): 1, (0,0): -1},
          {(0,0): 1, (0,1): 1},
          {(1,1): 1})),
        # s7 = (a12-a22)(b21+b22): c11 += s7
        (({(0,1): 1, (1,1): -1},
          {(1,0): 1, (1,1): 1},
          {(0,0): 1})),
    ]

    def add_4x4_strassen(b_col_off, c_col_off):
        """Add 49 Strassen terms for multiplying 4x4 A by 4x4 B block.
        
        b_col_off: column offset in B (0 or 4)
        c_col_off: column offset in C (0 or 4)
        """
        for outer_A, outer_B, outer_C in strassen_2x2:
            for inner_A, inner_B, inner_C in strassen_2x2:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                # Outer Strassen combines 2x2 blocks of A (row-wise) 
                # and 2x2 blocks of B (col-wise), contributing to C blocks.
                
                # For the outer term, outer_A tells us which A blocks are combined,
                # outer_B tells us which B blocks are combined.
                
                for (oor, ooc), c_ao in outer_A.items():
                    for (ior, ioc), c_ia in inner_A.items():
                        ar = oor * 2 + ior
                        ac = ooc * 2 + ioc
                        u[ar * 4 + ac] += c_ao * c_ia
                
                for (oor, ooc), c_bo in outer_B.items():
                    for (ior, ioc), c_ib in inner_B.items():
                        br = oor * 2 + ior
                        bc = b_col_off + ooc * 2 + ioc
                        v[br * 9 + bc] += c_bo * c_ib
                
                for (oor, ooc), c_co in outer_C.items():
                    for (ior, ioc), c_ic in inner_C.items():
                        cr = oor * 2 + ior
                        cc = c_col_off + ooc * 2 + ioc
                        w[cc * 4 + cr] += c_co * c_ic
                
                add_term(u, v, w)

    if time.time() > deadline:
        # Fallback: schoolbook
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0] * 16
                    v = [0.0] * 36
                    w = [0.0] * 36
                    u[i * 4 + j] = 1.0
                    v[j * 9 + k] = 1.0
                    w[k * 4 + i] = 1.0
                    add_term(u, v, w)
    else:
        # B split into B1(4x4, cols 0-3), B2(4x4, cols 4-7), B3(4x1, col 8)
        # 4x4 * B1: 49 terms
        add_4x4_strassen(0, 0)
        
        # 4x4 * B2: 49 terms
        add_4x4_strassen(4, 4)
        
        # 4x4 * B3: schoolbook, 16 terms
        for i in range(4):
            for j in range(4):
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                u[i * 4 + j] = 1.0
                v[j * 9 + 8] = 1.0
                w[8 * 4 + i] = 1.0
                add_term(u, v, w)
    
    rank = len(U_cols)
    
    # Build output matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(16)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(36)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}
