"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on each 2x2 block pair, splitting the 4 middle indices into
    two groups of 2. Total rank = 2 * 7 = 14 per output half, but since the two output
    columns of B are independent, we get 28 total.
    
    Actually, let me be more careful. A is 2x4, B is 4x4, C is 2x4.
    Split A's 4 columns into two 2-column blocks: A_L (2x2), A_R (2x2).
    Split B's 4 rows into two 2-row blocks: B_T (2x4), B_B (2x4).
    Split B's 4 columns into two 2-column blocks: B_L (4x2), B_R (4x2).
    
    C = A_L * B_T + A_R * B_B  (2x4 result)
    But B_T is 2x4, B_B is 2x4. So:
    C_left (2x2) = A_L * B_T[:, :2] + A_R * B_B[:, :2]  -- but B_T[:, :2] is 2x2, B_B[:, :2] is 2x2
    C_right (2x2) = A_L * B_T[:, 2:] + A_R * B_B[:, 2:]
    
    Each C_half is a sum of two 2x2 products. Since A_L, A_R, B_T[:, :2], B_B[:, :2] are all
    independent, the rank is 7 + 7 = 14. Similarly for the right half: 14. Total 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2x2 decomposition indices:
    # Given A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = [[c11,c12],[c21,c22]]
    # Strassen products:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # For each of the 4 (i, k_block) combinations where i in {0,1} (row of A) and 
    # k_block in {0,1} (which 2-column block of B), we use 7 Strassen terms.
    # Actually, the decomposition is per (A_block, B_block) pair.
    
    # A has 2 rows, 4 cols. Split cols into [0,1] and [2,3].
    # B has 4 rows, 4 cols. Split rows into [0,1] and [2,3], cols into [0,1] and [2,3].
    # C has 2 rows, 4 cols. Cols [0,1] come from B cols [0,1], cols [2,3] from B cols [2,3].
    
    # For C col-block b in {0,1}:
    #   C[:, 2b:2b+2] = A[:, 0:2] * B[0:2, 2b:2b+2] + A[:, 2:4] * B[2:4, 2b:2b+2]
    #   First term: 2x2 * 2x2, rank 7
    #   Second term: 2x2 * 2x2, rank 7
    #   Total per C col-block: 14
    #   Total: 28
    
    col = 0
    for b in range(2):  # B column block
        for a_blk in range(2):  # A column block / B row block
            # Strassen for this 2x2*2x2
            for s in range(7):
                c = col
                
                # U: 2x4 = 8 entries. A[i, j] where i in {0,1}, j depends on a_blk
                # B row block: rows 2*a_blk and 2*a_blk+1
                # B col block: cols 2*b and 2*b+1
                
                # Strassen combination for A (2x2): A[i, 2*a_blk + j]
                # For Strassen, the 2x2 A block has entries:
                #   a11 = A[0, 2*a_blk], a12 = A[0, 2*a_blk+1]
                #   a21 = A[1, 2*a_blk], a22 = A[1, 2*a_blk+1]
                
                # Strassen combination for B (2x2): B[2*a_blk+j, 2*b+k]
                #   b11 = B[2*a_blk, 2*b], b12 = B[2*a_blk, 2*b+1]
                #   b21 = B[2*a_blk+1, 2*b], b22 = B[2*a_blk+1, 2*b+1]
                
                # Strassen coefficients for A side (per product index s):
                # m1: (a11+a22) -> A[0,j0]+A[1,j1]
                # m2: (a21+a22)b11 -> (a21+a22) on A side
                # m3: a11 -> a11
                # m4: a22 -> a22
                # m5: (a11+a12) -> a11+a12
                # m6: (a21-a11) -> a21-a11
                # m7: (a12-a22) -> a12-a22
                
                # Let j0 = 2*a_blk, j1 = 2*a_blk+1
                j0 = 2 * a_blk
                j1 = 2 * a_blk + 1
                
                # A-side coefficients for each Strassen product
                a_coeffs = {
                    0: {0: 1, 1: 0, j0: 1, j1: 1},  # a11+a22: A[0,j0] + A[1,j1]
                    1: {0: 0, 1: 0, j0: 1, j1: 1},  # a21+a22: A[1,j0] + A[1,j1]
                    2: {0: 0, 1: 0, j0: 1},           # a11: A[0,j0]
                    3: {0: 0, 1: 0, j1: 1},           # a22: A[1,j1]
                    4: {0: 0, 1: 0, j0: 1, j1: 1},  # a11+a12: A[0,j0] + A[0,j1]
                    5: {0: 0, 1: 0, j0: -1, j1: 1},  # a21-a11: A[1,j0] - A[0,j0]... wait
                    6: {0: 0, 1: 0, j0: -1, j1: 1},  # a12-a22: A[0,j1] - A[1,j1]
                }
                
                # Let me redo the Strassen coefficients properly.
                # A block: A[i, 2*a_blk + j] for i in {0,1}, j in {0,1}
                #   a11 = A[0, 2*a_blk], a12 = A[0, 2*a_blk+1]
                #   a21 = A[1, 2*a_blk], a22 = A[1, 2*a_blk+1]
                #
                # Strassen products and A-side linear forms:
                # m1: a11 + a22 = A[0, 2*a_blk] + A[1, 2*a_blk+1]
                # m2: a21 + a22 = A[1, 2*a_blk] + A[1, 2*a_blk+1]
                # m3: a11 = A[0, 2*a_blk]
                # m4: a22 = A[1, 2*a_blk+1]
                # m5: a11 + a12 = A[0, 2*a_blk] + A[0, 2*a_blk+1]
                # m6: a21 - a11 = A[1, 2*a_blk] - A[0, 2*a_blk]
                # m7: a12 - a22 = A[0, 2*a_blk+1] - A[1, 2*a_blk+1]
                
                # B block: B[2*a_blk + j, 2*b + k] for j in {0,1}, k in {0,1}
                #   b11 = B[2*a_blk, 2*b], b12 = B[2*a_blk, 2*b+1]
                #   b21 = B[2*a_blk+1, 2*b], b22 = B[2*a_blk+1, 2*b+1]
                #
                # B-side linear forms:
                # m1: b11 + b22 = B[2*a_blk, 2*b] + B[2*a_blk+1, 2*b+1]
                # m2: b11 = B[2*a_blk, 2*b]
                # m3: b12 - b22 = B[2*a_blk, 2*b+1] - B[2*a_blk+1, 2*b+1]
                # m4: b21 - b11 = B[2*a_blk+1, 2*b] - B[2*a_blk, 2*b]
                # m5: b22 = B[2*a_blk+1, 2*b+1]
                # m6: b11 + b12 = B[2*a_blk, 2*b] + B[2*a_blk, 2*b+1]
                # m7: b21 + b22 = B[2*a_blk+1, 2*b] + B[2*a_blk+1, 2*b+1]
                
                # U side: A[i, 2*a_blk + j], i in {0,1}, j in {0,1}
                # U index: i * m + (2*a_blk + j) = i * 4 + 2*a_blk + j
                a_side = {
                    0: {0: 1, 1: 0},  # A[0,2a], A[1,2a+1] -> i=0,j0; i=1,j1
                    1: {0: 0, 1: 0, 2: 1, 3: 1},  # Hmm, let me use flat indices
                }
                
                # Let me use flat indices for U: row = i*4 + (2*a_blk + j)
                # i in {0,1}, j in {0,1} (local j within the 2-column block)
                # So U rows involved: 0*4 + 2*a_blk + 0, 0*4 + 2*a_blk + 1, 1*4 + 2*a_blk + 0, 1*4 + 2*a_blk + 1
                
                # For each Strassen product s, the A-side contributions:
                # s=0 (m1): A[0, 2a+0] + A[1, 2a+1] -> U[0*4+2a+0]=1, U[1*4+2a+1]=1
                # s=1 (m2): A[1, 2a+0] + A[1, 2a+1] -> U[1*4+2a+0]=1, U[1*4+2a+1]=1
                # s=2 (m3): A[0, 2a+0] -> U[0*4+2a+0]=1
                # s=3 (m4): A[1, 2a+1] -> U[1*4+2a+1]=1
                # s=4 (m5): A[0, 2a+0] + A[0, 2a+1] -> U[0*4+2a+0]=1, U[0*4+2a+1]=1
                # s=5 (m6): A[1, 2a+0] - A[0, 2a+0] -> U[1*4+2a+0]=1, U[0*4+2a+0]=-1
                # s=6 (m7): A[0, 2a+1] - A[1, 2a+1] -> U[0*4+2a+1]=1, U[1*4+2a+1]=-1
                
                U_contrib = {
                    0: [(0, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    1: [(1, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    2: [(0, 2*a_blk, 1)],
                    3: [(1, 2*a_blk+1, 1)],
                    4: [(0, 2*a_blk, 1), (0, 2*a_blk+1, 1)],
                    5: [(1, 2*a_blk, 1), (0, 2*a_blk, -1)],
                    6: [(0, 2*a_blk+1, 1), (1, 2*a_blk+1, -1)],
                }
                
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j][c] += coeff
                
                # V side: B[2*a_blk + j, 2*b + k], j in {0,1}, k in {0,1}
                # V index: (2*a_blk + j) * 4 + (2*b + k)
                
                V_contrib = {
                    0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],  # b11+b22
                    1: [(2*a_blk, 2*b, 1)],                           # b11
                    2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],  # b12-b22
                    3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],     # b21-b11
                    4: [(2*a_blk+1, 2*b+1, 1)],                        # b22
                    5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],       # b11+b12
                    6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],   # b21+b22
                }
                
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k][c] += coeff
                
                # W side: C[i, 2*b + k], i in {0,1}, k in {0,1}
                # W index: (2*b + k) * 2 + i (since p*n = 4*2 = 8, index = k_total * n + i = (2*b+k)*2 + i)
                
                # Strassen C reconstruction:
                # c11 = m1 + m4 - m5 + m7
                # c12 = m3 + m5
                # c21 = m2 + m4
                # c22 = m1 - m2 + m3 + m6
                
                # C local indices: (i_local, k_local) where i_local in {0,1}, k_local in {0,1}
                # C global: C[i_local, 2*b + k_local]
                # W index: (2*b + k_local) * 2 + i_local
                
                W_contrib = {
                    0: {(0, 0): 1, (1, 0): 1, (0, 1): -1, (1, 1): 1},  # m1: c11+=1, c21+=1; m4: c11+=1, c21+=1; m5: c11-=1, c21... 
                }
                
                # Let me be more careful. For each Strassen product s, its contribution to each C[i,k]:
                # s=0 (m1): c11 += 1, c22 += 1
                # s=1 (m2): c21 += 1, c22 -= 1
                # s=2 (m3): c12 += 1, c22 += 1
                # s=3 (m4): c11 += 1, c21 += 1
                # s=4 (m5): c11 -= 1, c12 += 1
                # s=5 (m6): c22 += 1
                # s=6 (m7): c11 += 1
                
                W_contrib = {
                    0: {(0, 0): 1, (1, 1): 1},       # m1: C[0,0]+=1, C[1,1]+=1
                    1: {(1, 0): 1, (1, 1): -1},      # m2: C[1,0]+=1, C[1,1]-=1
                    2: {(0, 1): 1, (1, 1): 1},       # m3: C[0,1]+=1, C[1,1]+=1
                    3: {(0, 0): 1, (1, 0): 1},       # m4: C[0,0]+=1, C[1,0]+=1
                    4: {(0, 0): -1, (0, 1): 1},      # m5: C[0,0]-=1, C[0,1]+=1
                    5: {(1, 1): 1},                   # m6: C[1,1]+=1
                    6: {(0, 0): 1},                   # m7: C[0,0]+=1
                }
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[(2 * b + k_loc) * 2 + i_loc][c] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
