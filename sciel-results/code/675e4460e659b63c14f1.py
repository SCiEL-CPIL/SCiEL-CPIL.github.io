"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    
    A is 2x4, B is 4x4, C is 2x4.
    Split A's 4 columns into two 2-column blocks: A_L (2x2), A_R (2x2).
    Split B's 4 rows into two 2-row blocks: B_T (2x4), B_B (2x4).
    Split B's 4 columns into two 2-column blocks: B_L (4x2), B_R (4x2).
    
    C[:, 0:2] = A_L * B_T[:, 0:2] + A_R * B_B[:, 0:2]
    C[:, 2:4] = A_L * B_T[:, 2:4] + A_R * B_B[:, 2:4]
    
    Each 2x2 * 2x2 product has rank 7 (Strassen), so each C half has rank 14.
    Total rank = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    col = 0
    for b in range(2):  # B column block: b=0 -> cols 0,1; b=1 -> cols 2,3
        for a_blk in range(2):  # A column block / B row block: a_blk=0 -> cols/rows 0,1; a_blk=1 -> cols/rows 2,3
            # Strassen for 2x2 * 2x2
            for s in range(7):
                c = col
                
                # A-side: A[i, 2*a_blk + j]
                # a11 = A[0, 2*a_blk], a12 = A[0, 2*a_blk+1]
                # a21 = A[1, 2*a_blk], a22 = A[1, 2*a_blk+1]
                #
                # Strassen A-side linear forms:
                # m1 (s=0): a11 + a22 = A[0, 2a] + A[1, 2a+1]
                # m2 (s=1): a21 + a22 = A[1, 2a] + A[1, 2a+1]
                # m3 (s=2): a11 = A[0, 2a]
                # m4 (s=3): a22 = A[1, 2a+1]
                # m5 (s=4): a11 + a12 = A[0, 2a] + A[0, 2a+1]
                # m6 (s=5): a21 - a11 = A[1, 2a] - A[0, 2a]
                # m7 (s=6): a12 - a22 = A[0, 2a+1] - A[1, 2a+1]
                
                j0 = 2 * a_blk
                j1 = 2 * a_blk + 1
                
                # U index: i * 4 + col_A where col_A is the actual column in A
                U_contrib = {
                    0: [(0, j0, 1), (1, j1, 1)],       # m1: A[0,2a] + A[1,2a+1]
                    1: [(1, j0, 1), (1, j1, 1)],       # m2: A[1,2a] + A[1,2a+1]
                    2: [(0, j0, 1)],                    # m3: A[0,2a]
                    3: [(1, j1, 1)],                    # m4: A[1,2a+1]
                    4: [(0, j0, 1), (0, j1, 1)],       # m5: A[0,2a] + A[0,2a+1]
                    5: [(1, j0, 1), (0, j0, -1)],      # m6: A[1,2a] - A[0,2a]
                    6: [(0, j1, 1), (1, j1, -1)],      # m7: A[0,2a+1] - A[1,2a+1]
                }
                
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j][c] += coeff
                
                # B-side: B[2*a_blk + j, 2*b + k]
                # b11 = B[2a, 2b], b12 = B[2a, 2b+1]
                # b21 = B[2a+1, 2b], b22 = B[2a+1, 2b+1]
                #
                # Strassen B-side linear forms:
                # m1 (s=0): b11 + b22 = B[2a,2b] + B[2a+1,2b+1]
                # m2 (s=1): b11 = B[2a,2b]
                # m3 (s=2): b12 - b22 = B[2a,2b+1] - B[2a+1,2b+1]
                # m4 (s=3): b21 - b11 = B[2a+1,2b] - B[2a,2b]
                # m5 (s=4): b22 = B[2a+1,2b+1]
                # m6 (s=5): b11 + b12 = B[2a,2b] + B[2a,2b+1]
                # m7 (s=6): b21 + b22 = B[2a+1,2b] + B[2a+1,2b+1]
                
                r0 = 2 * a_blk
                r1 = 2 * a_blk + 1
                k0 = 2 * b
                k1 = 2 * b + 1
                
                # V index: row_B * 4 + col_B
                V_contrib = {
                    0: [(r0, k0, 1), (r1, k1, 1)],       # m1: B[2a,2b] + B[2a+1,2b+1]
                    1: [(r0, k0, 1)],                    # m2: B[2a,2b]
                    2: [(r0, k1, 1), (r1, k1, -1)],      # m3: B[2a,2b+1] - B[2a+1,2b+1]
                    3: [(r1, k0, 1), (r0, k0, -1)],      # m4: B[2a+1,2b] - B[2a,2b]
                    4: [(r1, k1, 1)],                    # m5: B[2a+1,2b+1]
                    5: [(r0, k0, 1), (r0, k1, 1)],       # m6: B[2a,2b] + B[2a,2b+1]
                    6: [(r1, k0, 1), (r1, k1, 1)],       # m7: B[2a+1,2b] + B[2a+1,2b+1]
                }
                
                for (row_b, col_b, coeff) in V_contrib[s]:
                    V[row_b * 4 + col_b][c] += coeff
                
                # C-side: C[i, 2*b + k]
                # W index: (col_C) * n + i = (2*b + k) * 2 + i
                #
                # Strassen C reconstruction:
                # c11 = m1 + m4 - m5 + m7
                # c12 = m3 + m5
                # c21 = m2 + m4
                # c22 = m1 - m2 + m3 + m6
                #
                # Where c11 = C[0,0], c12 = C[0,1], c21 = C[1,0], c22 = C[1,1] (local)
                # Global: C[i_loc, 2*b + k_loc]
                # W index: (2*b + k_loc) * 2 + i_loc
                
                # For each Strassen product s, its contribution to each C[i_loc, k_loc]:
                # s=0 (m1): c11 += 1, c22 += 1  -> (0,0):+1, (1,1):+1
                # s=1 (m2): c21 += 1, c22 -= 1  -> (1,0):+1, (1,1):-1
                # s=2 (m3): c12 += 1, c22 += 1  -> (0,1):+1, (1,1):+1
                # s=3 (m4): c11 += 1, c21 += 1  -> (0,0):+1, (1,0):+1
                # s=4 (m5): c11 -= 1, c12 += 1  -> (0,0):-1, (0,1):+1
                # s=5 (m6): c22 += 1            -> (1,1):+1
                # s=6 (m7): c11 += 1            -> (0,0):+1
                
                W_contrib = {
                    0: [(0, 0, 1), (1, 1, 1)],       # m1: C[0,2b]+=1, C[1,2b+1]+=1
                    1: [(1, 0, 1), (1, 1, -1)],      # m2: C[1,2b]+=1, C[1,2b+1]-=1
                    2: [(0, 1, 1), (1, 1, 1)],       # m3: C[0,2b+1]+=1, C[1,2b+1]+=1
                    3: [(0, 0, 1), (1, 0, 1)],       # m4: C[0,2b]+=1, C[1,2b]+=1
                    4: [(0, 0, -1), (0, 1, 1)],      # m5: C[0,2b]-=1, C[0,2b+1]+=1
                    5: [(1, 1, 1)],                   # m6: C[1,2b+1]+=1
                    6: [(0, 0, 1)],                   # m7: C[0,2b]+=1
                }
                
                for (i_loc, k_loc, coeff) in W_contrib[s]:
                    W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
