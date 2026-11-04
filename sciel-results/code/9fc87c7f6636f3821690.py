"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # Split into row-blocks (size 2,1), col-blocks (size 2,2), out-blocks (size 2,2,2)
    row_blocks = [2, 1]
    col_blocks = [2, 2]
    out_blocks = [2, 2, 2]

    # For each (i_blk, j_blk, k_blk): rank = ri*rj*rk
    # Total: 2*2*2*2 + 2*2*2 + 1*2*2*2 + 1*2*2 = 16+8+8+4 = 36? Let me recount.
    # Actually: sum over I in {0,1}, J in {0,1}, K in {0,1,2}:
    #   |I|*|J|*|K| = 2*2*2 + 2*2*2 + 2*2*2 + 1*2*2 + 1*2*2 + 1*2*2 = 8+8+8+2+2+2 = 30
    # That's just schoolbook. We need Strassen for 2x2x2 blocks.
    # 2x2x2 blocks: (I=0,J=0,K=0), (I=0,J=1,K=0), (I=0,J=0,K=1), (I=0,J=1,K=1), (I=0,J=0,K=2), (I=0,J=1,K=2), (I=1,J=0,K=0), (I=1,J=1,K=0), (I=1,J=0,K=1), (I=1,J=1,K=1), (I=1,J=0,K=2), (I=1,J=1,K=2)
    # 2x2x2: I=0 (size 2), J in {0,1} (size 2), K in {0,1,2} (size 2) -> 6 blocks of 2x2x2 -> 6*7 = 42
    # 1x2x2: I=1 (size 1), J in {0,1} (size 2), K in {0,1,2} (size 2) -> 3 blocks of 1x2x2 -> 3*4 = 12
    # Total: 42 + 12 = 54. Still not great.
    #
    # Let me try a different split: row_blocks = [1, 2], so I=0 has size 1, I=1 has size 2.
    # 1x2x2 blocks: I=0, J in {0,1}, K in {0,1,2} -> 3 blocks, rank 4 each = 12
    # 2x2x2 blocks: I=1, J in {0,1}, K in {0,1,2} -> 6 blocks, rank 7 each = 42
    # Total: 54. Same.
    #
    # Try col_blocks = [1, 3]:
    # row_blocks=[2,1], col_blocks=[1,3], out_blocks=[2,2,2]
    # (I=0,J=0): 2x1x2 -> 3 blocks * rank 4 = 12
    # (I=0,J=1): 2x3x2 -> 3 blocks, each 2x1x2 -> 3*3*4 = 36
    # (I=1,J=0): 1x1x2 -> 3 blocks * rank 2 = 6
    # (I=1,J=1): 1x3x2 -> 3 blocks, each 1x1x2 -> 3*3*2 = 18
    # Total: 12+36+6+18 = 72. Worse.
    #
    # The key insight: 2x2x2 -> rank 7 (Strassen) saves 1 per block vs schoolbook rank 8.
    # 2x2x1 -> rank 4 = schoolbook, no savings.
    # 1x2x2 -> rank 4 = schoolbook, no savings.
    # 1x2x1 -> rank 1 = schoolbook, no savings.
    # So only 2x2x2 blocks give savings. Maximize number of 2x2x2 blocks.
    #
    # n=3, m=4, p=6. Best: row_blocks=[2,1], col_blocks=[2,2], out_blocks=[2,2,2]
    # 2x2x2: I=0 (size 2), J in {0,1} (size 2), K in {0,1,2} (size 2): 1*2*3 = 6 blocks, rank 7 each = 42
    # 1x2x2: I=1 (size 1), J in {0,1} (size 2), K in {0,1,2} (size 2): 1*2*3 = 6 blocks, rank 4 each = 24
    # Total: 42 + 24 = 66. Slightly better than 68.
    #
    # Can we do better? What if we use a non-rectangular decomposition?
    # Actually, let me just implement the block approach carefully.
    #
    # For 2x2x2 Strassen (computing C = A*B where A,B,C are 2x2):
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    #
    # Each M is a product of a linear combo of A-entries and a linear combo of B-entries.
    # In the tensor decomposition, each Strassen term contributes to multiple (i,j,k) entries.
    #
    # For a 2x2x2 block at position (i0, j0, k0) [offsets], the 7 Strassen terms contribute:
    # Term t has U-part: u_i[i'] for i' in {i0, i0+1}, V-part: v_j[j'] for j' in {j0, j0+1},
    # and W-part: w_k[k'] for k' in {k0, k0+1}.
    #
    # Let me define the Strassen coefficients. For local indices (a,b) in A, (c,d) in B, (e,f) in C:
    # M1: A coeff: {(0,0):1, (1,1):1}, B coeff: {(0,0):1, (1,1):1}, C coeff: {(0,0):1, (1,1):1}
    # M2: A coeff: {(1,0):1, (1,1):1}, B coeff: {(0,0):1}, C coeff: {(1,0):1}
    # M3: A coeff: {(0,0):1}, B coeff: {(0,1):1, (1,1):-1}, C coeff: {(0,1):1}
    # M4: A coeff: {(1,1):1}, B coeff: {(1,0):1, (0,0):-1}, C coeff: {(0,0):1, (1,0):1}
    # M5: A coeff: {(0,0):1, (0,1):1}, B coeff: {(1,1):1}, C coeff: {(0,0):-1, (1,1):1}
    # M6: A coeff: {(1,0):1, (0,0):-1}, B coeff: {(0,0):1, (0,1):1}, C coeff: {(1,1):1}
    # M7: A coeff: {(0,1):1, (1,1):-1}, B coeff: {(1,0):1, (1,1):1}, C coeff: {(0,0):1, (1,1):-1}

    # Strassen decomposition for 2x2x2
    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: list of (local_i, local_j, val)
        # V_coeffs: list of (local_j, local_k, val)
        # W_coeffs: list of (local_k, local_i, val)
        [([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])],
        [([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)])],
        [([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)])],
        [([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])],
        [([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,1,1)])],
        [([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])],
        [([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,-1)])],
    ]

    # Direct rank decomposition for 1x2x2 (I has size 1, J has size 2, K has size 2): rank 4
    # For local i=0, j in {0,1}, k in {0,1}: just schoolbook, 4 terms
    # Each term: U[(0,j)], V[(j,k)], W[(k,0)]
    # Actually for 1x2x2, rank = 1*2*2 = 4, which is just schoolbook.

    # Build the full decomposition
    # Row blocks: I=0 has rows {0,1}, I=1 has row {2}
    # Col blocks: J=0 has cols {0,1}, J=1 has cols {2,3}
    # Out blocks: K=0 has out {0,1}, K=1 has out {2,3}, K=2 has out {4,5}

    U_entries = []  # list of (row_idx, col_idx, val)
    V_entries = []
    W_entries = []

    col = 0
    for I in range(2):  # row block index
        i0 = 0 if I == 0 else 2
        ri = 2 if I == 0 else 1
        for J in range(2):  # col block index
            j0 = 0 if J == 0 else 2
            rj = 2  # both col blocks have size 2
            for K in range(3):  # out block index
                k0 = 2 * K
                rk = 2  # both out blocks have size 2

                if ri == 2 and rj == 2 and rk == 2:
                    # 2x2x2: use Strassen, rank 7
                    for term in strassen:
                        u_coeffs, v_coeffs, w_coeffs = term
                        # Add to U, V, W
                        for (li, lj, val) in u_coeffs:
                            row_idx = (i0 + li) * m + (j0 + lj)
                            U_entries.append((row_idx, col, val))
                        for (lj, lk, val) in v_coeffs:
                            row_idx = (j0 + lj) * p + (k0 + lk)
                            V_entries.append((row_idx, col, val))
                        for (lk, li, val) in w_coeffs:
                            row_idx = (k0 + lk) * n + (i0 + li)
                            W_entries.append((row_idx, col, val))
                        col += 1
                elif ri == 1 and rj == 2 and rk == 2:
                    # 1x2x2: rank 4, schoolbook
                    for lj in range(rj):
                        for lk in range(rk):
                            row_idx_u = (i0) * m + (j0 + lj)
                            row_idx_v = (j0 + lj) * p + (k0 + lk)
                            row_idx_w = (k0 + lk) * n + (i0)
                            U_entries.append((row_idx_u, col, 1))
                            V_entries.append((row_idx_v, col, 1))
                            W_entries.append((row_idx_w, col, 1))
                            col += 1
                else:
                    # Fallback: schoolbook for other sizes
                    for li in range(ri):
                        for lj in range(rj):
                            for lk in range(rk):
                                row_idx_u = (i0 + li) * m + (j0 + lj)
                                row_idx_v = (j0 + lj) * p + (k0 + lk)
                                row_idx_w = (k0 + lk) * n + (i0 + li)
                                U_entries.append((row_idx_u, col, 1))
                                V_entries.append((row_idx_v, col, 1))
                                W_entries.append((row_idx_w, col, 1))
                                col += 1

    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for (row, c, val) in U_entries:
        U[row][c] = val
    for (row, c, val) in V_entries:
        V[row][c] = val
    for (row, c, val) in W_entries:
        W[row][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
