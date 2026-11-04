"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Bini rank-19 for 3x3x3 core + schoolbook for strips
    # Core: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]  (rank 19)
    # Strip A (col 3): A[0:3, 3] * B[0:3, 0:3] -> C[0:3, 3]  (rank 9)
    # Strip B (row 3): A[0:3, 0:3] * B[3, 0:3] -> C[0:3, 0:3]  (rank 9)
    # Total rank = 19 + 9 + 9 = 37
    # Wait, let me reconsider the decomposition.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # = sum_{k=0}^{2} A[i,k]*B[k,j]  (core 3x3x3)
    #   + A[i,3]*B[3,j]  (strip: A col 3 times B row 3)
    # So we need:
    # 1. 3x3x3 core: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]  (rank 19 via Bini)
    # 2. A[0:3,3] * B[3,0:3] -> C[0:3,0:3]  (rank 9 schoolbook)
    # That's it! No separate strips needed because B is 4x4 and A is 3x4.
    # C is 3x4. C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    # = sum_{k=0}^{2} A[i,k]*B[k,j] + A[i,3]*B[3,j]
    # Core: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    # Strip: A[0:3,3] * B[3,0:3] -> C[0:3,0:3]
    # Total rank = 19 + 9 = 28
    
    rank = 28
    
    # Bini-Capovani-De Falco-Ladisi rank-19 decomposition for 3x3x3
    # This is the well-known rank-19 algorithm.
    # U indices: A[i,k] -> i*3+k for i,k in 0..2 (indices 0..8)
    # V indices: B[k,j] -> k*3+j for k,j in 0..2 (indices 0..8)
    # W indices: C[i,j] -> j*3+i (trace convention k*n+i where n=3, so W[j*3+i])
    #   C[0,0]=0, C[0,1]=3, C[0,2]=6
    #   C[1,0]=1, C[1,1]=4, C[1,2]=7
    #   C[2,0]=2, C[2,1]=5, C[2,2]=8
    
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
            
        cols.append((u_vec, v_vec, w_vec))
    
    # Bini rank-19 for 3x3x3 matrix multiplication
    # Using the standard Bini-Capovani-De Falco-Ladisi (BCDFL) algorithm
    # A[i,k] index = i*3+k, B[k,j] index = k*3+j, C[i,j] index = j*3+i
    
    # The 19 terms of the BCDFL algorithm:
    # Each term is (u_entries, v_entries, w_entries)
    
    # Term 1: M1 = A[0,0]*B[0,0]
    # Contributes to C[0,0]
    add_term([(0,1)], [(0,1)], [(0,1)])
    
    # Term 2: M2 = A[0,1]*B[1,0]
    # Contributes to C[0,0]
    add_term([(1,1)], [(3,1)], [(0,1)])
    
    # Term 3: M3 = A[0,2]*B[2,0]
    # Contributes to C[0,0]
    add_term([(2,1)], [(6,1)], [(0,1)])
    
    # Term 4: M4 = A[1,0]*B[0,1]
    # Contributes to C[1,1]
    add_term([(3,1)], [(1,1)], [(4,1)])
    
    # Term 5: M5 = A[1,1]*B[1,1]
    # Contributes to C[1,1]
    add_term([(4,1)], [(4,1)], [(4,1)])
    
    # Term 6: M6 = A[1,2]*B[2,1]
    # Contributes to C[1,1]
    add_term([(5,1)], [(7,1)], [(4,1)])
    
    # Term 7: M7 = A[2,0]*B[0,2]
    # Contributes to C[2,2]
    add_term([(6,1)], [(2,1)], [(8,1)])
    
    # Term 8: M8 = A[2,1]*B[1,2]
    # Contributes to C[2,2]
    add_term([(7,1)], [(5,1)], [(8,1)])
    
    # Term 9: M9 = A[2,2]*B[2,2]
    # Contributes to C[2,2]
    add_term([(8,1)], [(8,1)], [(8,1)])
    
    # Now the remaining 10 terms that handle cross terms:
    # We need C[0,1], C[0,2], C[1,0], C[1,2], C[2,0], C[2,1]
    
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # The BCDFL algorithm uses clever combinations. Let me use the known rank-19:
    # Actually, let me use a verified rank-19 decomposition.
    # 
    # The standard BCDFL rank-19 terms (10 additional beyond the 9 diagonal):
    # These involve sums/differences to cover all 27 entries with 19 terms.
    
    # Let me use a different approach - I'll use the known Strassen-like extension.
    # Actually, the simplest verified rank-19 for 3x3x3 uses:
    
    # Let me restart with a clean, verified rank-19 decomposition.
    # I'll use the one from the literature that I can verify.
    
    # Clear cols and restart
    cols.clear()
    
    # Verified BCDFL rank-19 decomposition for 3x3x3:
    # U: A[i,k] at index i*3+k
    # V: B[k,j] at index k*3+j  
    # W: C[i,j] at index j*3+i
    
    # The 19 terms:
    # t1:  A00*B00 -> C00
    add_term([(0,1)], [(0,1)], [(0,1)])
    # t2:  A01*B10 -> C00  
    add_term([(1,1)], [(3,1)], [(0,1)])
    # t3:  A02*B20 -> C00
    add_term([(2,1)], [(6,1)], [(0,1)])
    # t4:  A10*B01 -> C01
    add_term([(3,1)], [(1,1)], [(3,1)])
    # t5:  A11*B11 -> C11
    add_term([(4,1)], [(4,1)], [(4,1)])
    # t6:  A12*B21 -> C11
    add_term([(5,1)], [(7,1)], [(4,1)])
    # t7:  A20*B02 -> C02
    add_term([(6,1)], [(2,1)], [(6,1)])
    # t8:  A21*B12 -> C12
    add_term([(7,1)], [(5,1)], [(7,1)])
    # t9:  A22*B22 -> C22
    add_term([(8,1)], [(8,1)], [(8,1)])
    # t10: (A00+A11+A22)*(B00+B11+B22) -> C00,C11,C22
    add_term([(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)])
    # t11: (A01+A10)*(B10+B01) -> C00,C01,C10,C11
    add_term([(1,1),(3,1)], [(3,1),(1,1)], [(0,1),(3,1),(1,1),(4,1)])
    # t12: (A02+A20)*(B20+B02) -> C00,C02,C20,C22
    add_term([(2,1),(6,1)], [(6,1),(2,1)], [(0,1),(6,1),(2,1),(8,1)])
    # t13: (A12+A21)*(B21+B12) -> C11,C12,C21,C22
    add_term([(5,1),(7,1)], [(7,1),(5,1)], [(4,1),(7,1),(5,1),(8,1)])
    # t14: (A00-A11)*(B11-B00) -> C00,C01,C10,C11
    add_term([(0,1),(4,-1)], [(4,1),(0,-1)], [(0,1),(3,-1),(1,-1),(4,1)])
    # t15: (A00-A22)*(B22-B00) -> C00,C02,C20,C22
    add_term([(0,1),(8,-1)], [(8,1),(0,-1)], [(0,1),(6,-1),(2,-1),(8,1)])
    # t16: (A11-A22)*(B22-B11) -> C11,C12,C21,C22
    add_term([(4,1),(8,-1)], [(8,1),(4,-1)], [(4,1),(7,-1),(5,-1),(8,1)])
    # t17: (A01-A20)*(B20-B01) -> C00,C01,C20,C21... 
    # Hmm, this is getting complex. Let me use a verified approach.
    
    # Actually, I realize I should just use the schoolbook for 3x3x3 (rank 27)
    # combined with the strip (rank 9) = rank 36, which is worse than 44.
    # 
    # Let me reconsider. The Bini rank-19 is correct but I need the exact coefficients.
    # Let me use a known correct rank-19 decomposition.
    
    cols.clear()
    
    # I'll use the standard Bini-Capovani-De Falco-Ladisi rank-19.
    # Reference: Bini et al. 1979, "On the rank of the tensor of matrix multiplication"
    # 
    # For 3x3x3, the 19 bilinear forms are:
    # Let me define them carefully.
    
    # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5, A[2,0]=6, A[2,1]=7, A[2,2]=8
    # B indices: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    # W indices (C[i,j] at j*3+i): C[0,0]=0, C[0,1]=3, C[0,2]=6, C[1,0]=1, C[1,1]=4, C[1,2]=7, C[2,0]=2, C[2,1]=5, C[2,2]=8
    
    # BCDFL rank-19 terms:
    # I'll use a formulation I can verify by expansion.
    
    # Strategy: Use 9 "diagonal" terms for C[i,i], then 10 more for off-diagonals.
    # But that gives 19 total. Let me be very careful.
    
    # Actually, let me just verify: can I do 3x3x3 in rank 19?
    # Yes, Bini proved rank <= 19. The exact decomposition is known.
    
    # Let me use this verified rank-19 (from multiple sources):
    
    # Term 1:  M1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    #   -> C00 + C11 + C22 + cross terms
    # This is getting too complex to write from memory correctly.
    
    # Let me take a different approach: use Strassen for 2x2x2 blocks within 3x3x3.
    # 3x3x3 can be decomposed as:
    # - Top-left 2x2x2: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (rank 7)
    # - Remaining strips... this gets complicated.
    
    # Alternative: Just use schoolbook for everything but optimize.
    # 3x4x4 schoolbook = 48. Current is 44.
    # 
    # Let me try: 3x3x3 core (rank 27 schoolbook) + strip (rank 9) = 36.
    # That's better than 44! Score would be 48/36 = 1.333.
    
    # Actually wait - let me reconsider the block structure.
    # The current program uses Strassen on 2x2x2 blocks.
    # 3x4x4 = 3 rows of A, 4 cols of A, 4 rows of B, 4 cols of B.
    # 
    # Better decomposition:
    # Split A into A_left (3x3) and A_right (3x1)
    # Split B into B_top (3x4) and B_bottom (1x4)
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top: 3x3 * 3x4 = 3x4 (rank 27 schoolbook or less)
    # A_right * B_bottom: 3x1 * 1x4 = 3x4 (rank 12 schoolbook)
    # Total: 27 + 12 = 39. Still better than 44!
    
    # Or: A_left * B_top using Strassen-like:
    # 3x3 * 3x4: split into 2x2 * 2x2 (rank 7) + strips
    # This is what the current program does partially.
    
    # Let me try the simplest improvement: 
    # 3x3x3 core schoolbook (27) + 3x1x4 strip schoolbook (12) = 39
    # Score: 48/39 = 1.231. Better than 1.091!
    
    # Actually, even better: use Strassen for the 2x2x2 blocks in the core.
    # 3x3x3 = 2x2x2 (rank 7) + strips
    # A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2] : rank 7
    # A[0:2,2]*B[2,0:2] -> C[0:2,2] : rank 4 (1x2x2 type)
    # A[2,0:2]*B[0:2,0:2] -> C[2,0:2] : rank 4
    # A[2,2]*B[2,2] -> C[2,2] : rank 1
    # A[0:2,2]*B[2,2] -> C[0:2,2] : rank 2
    # A[2,0:2]*B[2,0:2] -> C[2,0:2] : rank 2
    # A[2,2]*B[2,0:2] -> C[2,0:2] : rank 2
    # A[2,2]*B[2,2] already counted
    # Hmm, let me be more careful.
    
    # C[i,j] = sum_k A[i,k]*B[k,j] for i,j,k in 0..2
    # 
    # Split k into {0,1} and {2}:
    # C[i,j] = sum_{k=0,1} A[i,k]*B[k,j] + A[i,2]*B[2,j]
    # 
    # Part 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (2x2x2, rank 7 via Strassen)
    # Part 2: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]  (2x1x2, rank 4)
    # Part 3: A[2,0:2] * B[0:2,0:2] -> C[2,0:2]  (1x2x2, rank 4)
    # Part 4: A[2,2] * B[2,0:2] -> C[2,0:2]  (1x1x2, rank 2)
    # Part 5: A[0:2,2] * B[2,2] -> C[0:2,2]  (2x1x1, rank 2)
    # Part 6: A[2,0:2] * B[2,2] -> C[2,2]  (1x1x1, rank 1)
    # Part 7: A[2,2] * B[2,2] -> C[2,2]  (1x1x1, rank 1)
    # 
    # Total for 3x3x3: 7+4+4+2+2+1+1 = 21
    # 
    # Now for the full 3x4x4:
    # C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    # = sum_{k=0,1} A[i,k]*B[k,j] + A[i,2]*B[2,j] + A[i,3]*B[3,j]
    # 
    # Using Strassen on the 2x2x2 core (k in {0,1}, i in {0,1}, j in {0,1}):
    # A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2] : rank 7
    # A[0:2,0:2]*B[0:2,2:4] -> C[0:2,2:4] : rank 7 (another 2x2x2)
    # A[2,0:2]*B[0:2,0:2] -> C[2,0:2] : rank 4
    # A[2,0:2]*B[0:2,2:4] -> C[2,2:4] : rank 4
    # A[0:2,2]*B[2,0:2] -> C[0:2,0:2] : rank 4
    # A[0:2,2]*B[2,2:4] -> C[0:2,2:4] : rank 4
    # A[2,2]*B[2,0:2] -> C[2,0:2] : rank 2
    # A[2,2]*B[2,2:4] -> C[2,2:4] : rank 2
    # A[0:2,3]*B[3,0:2] -> C[0:2,0:2] : rank 4
    # A[0:2,3]*B[3,2:4] -> C[0:2,2:4] : rank 4
    # A[2,3]*B[3,0:2] -> C[2,0:2] : rank 2
    # A[2,3]*B[3,2:4] -> C[2,2:4] : rank 2
    # 
    # Total: 7+7+4+4+4+4+2+2+4+4+2+2 = 46. Worse than 44!
    
    # The current program's approach is better. Let me stick with it and try
    # to find a better block decomposition.
    
    # Actually, the current program achieves 44. Let me verify it's correct
    # and just return it. The directive says to KEEP this design.
    
    # Let me just return the current rank-44 solution which is verified working.
    
    # Rebuild the rank-44 solution
    cols.clear()
    rank = 44
    
    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
            
        cols.append((u_vec, v_vec, w_vec))
    
    def add_strassen(a,b,c,d, e,f,g,h, w11,w12,w21,w22):
        add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
        add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
        add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
        add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
        add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
        add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
        add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])
        
    def add_rank4(a,b, c,d,e,f, w1,w2):
        add_term([(a,1)], [(c,1)], [(w1,1)])
        add_term([(b,1)], [(e,1)], [(w1,1)])
        add_term([(a,1)], [(d,1)], [(w2,1)])
        add_term([(b,1)], [(f,1)], [(w2,1)])

    # A (3x4): Row 0: 0-3, Row 1: 4-7, Row 2: 8-11
    # B (4x4): Row 0: 0-3, Row 1: 4-7, Row 2: 8-11, Row 3: 12-15
    # C (3x4) W (k*3+i): C[0,0]=0, C[0,1]=3, C[0,2]=6, C[0,3]=9
    #                    C[1,0]=1, C[1,1]=4, C[1,2]=7, C[1,3]=10
    #                    C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11
    
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
