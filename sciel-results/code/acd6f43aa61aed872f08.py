"""Matrix multiplication 4,4,9 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit (5s safety margin)
    
    n, m, p = DIMENSIONS
    # Rank for 4x4*4x9 using block decomposition with Strassen
    # Split into 2x2 blocks of A (4 blocks), B has 4x(4x2+1) blocks
    # Each A_block (2x2) * B_block:
    #   4 B_blocks of 2x2: rank 7 each (Strassen) = 28
    #   1 B_block of 2x1: rank 4 (schoolbook) = 4
    # Total per A_block: 32
    # 4 A_blocks: 4 * 32 = 128
    rank = 128
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 128
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 128
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 128
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e), M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We decompose 4x4 * 4x9
    # A is 4x4, split into four 2x2 blocks: A00, A01, A10, A11
    # B is 4x9, split into blocks:
    #   Row blocks: 2 blocks of 2 rows
    #   Column blocks: 4 blocks of 2 cols + 1 block of 1 col
    # So B has 2x5 = 10 blocks: B_ij where i in {0,1}, j in {0,1,2,3,4}
    # B_ij is 2x2 for j<4, 2x1 for j=4
    
    # For each A_block (2x2) and each B_block:
    #   If B_block is 2x2: use Strassen (rank 7)
    #   If B_block is 2x1: use schoolbook (rank 4)
    
    # Let's index the blocks
    # A blocks: A_ab where a in {0,1} (row block), b in {0,1} (col block)
    # B blocks: B_cd where c in {0,1} (row block), d in {0,1,2,3,4} (col block)
    # C blocks: C_ad where a in {0,1}, d in {0,1,2,3,4}
    
    # For each pair (A_ab, B_bc) where b is the inner index:
    # Actually, C_ad = sum_b A_ab * B_bd
    # So for each a in {0,1}, d in {0,1,2,3,4}:
    #   C_ad = sum_{b in {0,1}} A_ab * B_bd
    
    # Let me restructure. For each a in {0,1} (row block of A/C) and d in {0,1,2,3,4} (col block of B/C):
    #   We need to compute C_ad = A_a0 * B_0d + A_a1 * B_1d
    #   where A_a0 is 2x2, A_a1 is 2x2, B_0d is 2x2 or 2x1, B_1d is 2x2 or 2x1
    
    # If d < 4: B_0d and B_1d are 2x2
    #   C_ad = A_a0 * B_0d + A_a1 * B_1d
    #   This is sum of two 2x2*2x2 multiplications
    #   Each 2x2*2x2 has rank 7 (Strassen)
    #   So total rank for this C_ad is 14? No, we can share some structure.
    #   Actually, C_ad is 2x2, computed as sum of two rank-7 tensors.
    #   The tensor for A_a0 * B_0d has rank 7, and A_a1 * B_1d has rank 7.
    #   These are independent, so total rank is 14.
    
    # If d = 4: B_0d and B_1d are 2x1
    #   C_ad = A_a0 * B_0d + A_a1 * B_1d
    #   Each 2x2*2x1 has rank 4 (schoolbook)
    #   Total rank for this C_ad is 8.
    
    # Total rank:
    #   For d in {0,1,2,3}: 4 blocks, each with a in {0,1}, so 8 C_blocks of rank 14 = 112
    #   For d = 4: 2 C_blocks of rank 8 = 16
    #   Total: 112 + 16 = 128
    
    # Now let's build the factor matrices
    # We'll assign column indices
    col = 0
    for a in range(2):  # row block of A/C
        for d in range(5):  # col block of B/C
            if d < 4:
                # Two 2x2 * 2x2 multiplications, each rank 7
                for b in range(2):  # inner sum index
                    # A_ab * B_bd, both 2x2
                    # Strassen rank 7
                    for s in range(7):
                        cidx = col
                        # A_ab indices: a*2 + r, b*2 + c for r,c in {0,1}
                        # In global A indices: row = a*2 + r, col = b*2 + c
                        # In U indices: row * m + col = (a*2+r)*4 + (b*2+c)
                        # B_bd indices: row = b*2 + r', col = d*2 + c' for r',c' in {0,1}
                        # In global B indices: row = b*2 + r', col = d*2 + c'
                        # In V indices: row * p + col = (b*2+r')*9 + (d*2+c')
                        # C_ad indices: row = a*2 + r'', col = d*2 + c'' for r'',c'' in {0,1}
                        # In global C indices: row = a*2 + r'', col = d*2 + c''
                        # In W indices: col * n + row = (d*2+c'')*4 + (a*2+r'')
                        
                        # Strassen formulas:
                        # M1 = (a+d)(e+h): A[0,0]+A[1,1] times B[0,0]+B[1,1]
                        # M2 = (c+d)e: A[1,0]+A[1,1] times B[0,0]
                        # M3 = a(f-h): A[0,0] times B[0,1]-B[1,1]
                        # M4 = d(g-e): A[1,1] times B[1,0]-B[0,0]
                        # M5 = (a+b)h: A[0,0]+A[0,1] times B[1,1]
                        # M6 = (c-a)(e+f): A[1,0]-A[0,0] times B[0,0]+B[0,1]
                        # M7 = (b-d)(g+h): A[0,1]-A[1,1] times B[1,0]+B[1,1]
                        
                        # For M_s, U factors are coefficients on A entries
                        # V factors are coefficients on B entries
                        # W factors are coefficients on C entries
                        
                        # M1: U = (1,0,0,1) on A[00],A[01],A[10],A[11]
                        #     V = (1,0,0,1) on B[00],B[01],B[10],B[11]
                        #     W: C11 += M1, C12 -= 0, C21 -= 0, C22 -= M1
                        #     W = (1,0,0,-1) on C[00],C[01],C[10],C[11]
                        if s == 0:  # M1
                            U[(a*2+0)*4 + (b*2+0)][cidx] += 1
                            U[(a*2+1)*4 + (b*2+1)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+0)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+1)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+0)][cidx] += 1
                            W[(d*2+1)*4 + (a*2+1)][cidx] += -1
                        elif s == 1:  # M2
                            U[(a*2+1)*4 + (b*2+0)][cidx] += 1
                            U[(a*2+1)*4 + (b*2+1)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+0)][cidx] += 1
                            W[(d*2+1)*4 + (a*2+1)][cidx] += 1
                            W[(d*2+1)*4 + (a*2+0)][cidx] += 1
                        elif s == 2:  # M3
                            U[(a*2+0)*4 + (b*2+0)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+1)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+1)][cidx] += -1
                            W[(d*2+0)*4 + (a*2+1)][cidx] += 1
                            W[(d*2+1)*4 + (a*2+0)][cidx] += 1
                        elif s == 3:  # M4
                            U[(a*2+1)*4 + (b*2+1)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+0)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+0)][cidx] += -1
                            W[(d*2+0)*4 + (a*2+0)][cidx] += 1
                            W[(d*2+1)*4 + (a*2+1)][cidx] += 1
                        elif s == 4:  # M5
                            U[(a*2+0)*4 + (b*2+0)][cidx] += 1
                            U[(a*2+0)*4 + (b*2+1)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+1)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+0)][cidx] += -1
                            W[(d*2+0)*4 + (a*2+1)][cidx] += 1
                        elif s == 5:  # M6
                            U[(a*2+1)*4 + (b*2+0)][cidx] += 1
                            U[(a*2+0)*4 + (b*2+0)][cidx] += -1
                            V[(b*2+0)*9 + (d*2+0)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+1)][cidx] += 1
                            W[(d*2+1)*4 + (a*2+1)][cidx] += 1
                        elif s == 6:  # M7
                            U[(a*2+0)*4 + (b*2+1)][cidx] += 1
                            U[(a*2+1)*4 + (b*2+1)][cidx] += -1
                            V[(b*2+1)*9 + (d*2+0)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+1)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+0)][cidx] += 1
                        col += 1
            else:
                # d = 4: B_bd is 2x1 for b in {0,1}
                # C_ad = A_a0 * B_04 + A_a1 * B_14
                # Each 2x2 * 2x1 has rank 4 (schoolbook)
                # C_ad is 2x1
                for b in range(2):
                    # A_ab * B_bd where A_ab is 2x2, B_bd is 2x1
                    # C_r = sum_c A[r,c] * B[c,0]
                    # C_0 = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C_1 = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                    # Rank 4: 4 terms
                    for t in range(4):
                        cidx = col
                        if t == 0:
                            U[(a*2+0)*4 + (b*2+0)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+0)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+0)][cidx] += 1
                        elif t == 1:
                            U[(a*2+0)*4 + (b*2+1)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+0)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+0)][cidx] += 1
                        elif t == 2:
                            U[(a*2+1)*4 + (b*2+0)][cidx] += 1
                            V[(b*2+0)*9 + (d*2+0)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+1)][cidx] += 1
                        elif t == 3:
                            U[(a*2+1)*4 + (b*2+1)][cidx] += 1
                            V[(b*2+1)*9 + (d*2+0)][cidx] += 1
                            W[(d*2+0)*4 + (a*2+1)][cidx] += 1
                        col += 1
    
    # Check that we used exactly rank columns
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
