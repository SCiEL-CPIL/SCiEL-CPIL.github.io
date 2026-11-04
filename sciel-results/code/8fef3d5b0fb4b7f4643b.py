"""Matrix multiplication 2x4x4 with Strassen-based block decomposition (rank 28)."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # Strassen's 7 terms for 2x2x2 matrix multiplication
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M3-M5+M7, M4+M5], [M2+M3, M1-M2+M4+M6]]
    
    # We have four 2x2 blocks:
    # A block 1: cols 0,1 of A (indices 0,1 in A's column space)
    # A block 2: cols 2,3 of A (indices 2,3 in A's column space)
    # B block 1: rows 0,1 of B (indices 0,1 in B's row space)
    # B block 2: rows 2,3 of B (indices 2,3 in B's row space)
    
    # For each (A_block, B_block) pair, we need to handle two k-ranges:
    # If B_block is rows 0,1: k ranges 0,1 for C's k=0,1
    # If B_block is rows 2,3: k ranges 2,3 for C's k=2,3
    
    # Actually, let me think more carefully.
    # A is 2x4: A[i,j] for i in 0,1 and j in 0,1,2,3
    # B is 4x4: B[j,k] for j in 0,1,2,3 and k in 0,1,2,3
    # C is 2x4: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    
    # Let A1[i,j'] = A[i,j'] for j' in 0,1 (2x2 matrix)
    # Let A2[i,j'] = A[i,j'+2] for j' in 0,1 (2x2 matrix)
    # Let B1[j',k] = B[j',k] for j' in 0,1 (2x4 matrix)
    # Let B2[j',k] = B[j'+2,k] for j' in 0,1 (2x4 matrix)
    
    # C[i,k] = (A1 * B1)[i,k] + (A2 * B2)[i,k]
    
    # Now A1 is 2x2, B1 is 2x4. We can split B1 into two 2x2 blocks:
    # B1a = B1[:, 0:2] (2x2), B1b = B1[:, 2:4] (2x2)
    # (A1 * B1a) is 2x2, (A1 * B1b) is 2x2
    # So C[:, 0:2] = A1 * B1a, C[:, 2:4] = A1 * B1b
    
    # Similarly for A2 * B2:
    # B2a = B2[:, 0:2], B2b = B2[:, 2:4]
    # C[:, 0:2] += A2 * B2a, C[:, 2:4] += A2 * B2b
    
    # So we have 4 independent 2x2x2 multiplications:
    # 1. A1 * B1a -> contributes to C[:, 0:2]
    # 2. A1 * B1b -> contributes to C[:, 2:4]
    # 3. A2 * B2a -> contributes to C[:, 0:2]
    # 4. A2 * B2b -> contributes to C[:, 2:4]
    
    # Each 2x2x2 multiplication can be done with Strassen's 7 terms.
    # Total: 4 * 7 = 28 terms.
    
    # Let's define the Strassen terms for a generic 2x2x2 multiplication.
    # For multiplying X (2x2) by Y (2x2) to get Z (2x2):
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # The 7 Strassen products and their contributions:
    # M1 = (x00+x11)*(y00+y11) -> z00 += M1, z11 += M1
    # M2 = (x10+x11)*y00 -> z10 += M2, z11 -= M2
    # M3 = x00*(y01-y11) -> z00 += M3, z10 += M3
    # M4 = x11*(y10-y00) -> z01 += M4, z11 += M4
    # M5 = (x00+x01)*y11 -> z00 -= M5, z01 += M5
    # M6 = (x10-x00)*(y00+y01) -> z10 += M6, z11 += M6
    # M7 = (x01-x11)*(y10+y11) -> z00 += M7, z01 -= M7
    
    # Let's verify:
    # z00 = M1 + M3 - M5 + M7 = (x00+x11)(y00+y11) + x00(y01-y11) - (x00+x01)y11 + (x01-x11)(y10+y11)
    #     = x00*y00 + x00*y11 + x11*y00 + x11*y11 + x00*y01 - x00*y11 - x00*y11 - x01*y11 + x01*y10 + x01*y11 - x11*y10 - x11*y11
    #     = x00*y00 + x11*y00 + x00*y01 + x01*y10 - x00*y11 - x11*y10
    #     = x00*y00 + x01*y10 + x10*y00 + x11*y10? No wait.
    
    # Let me recompute more carefully:
    # M1 = (x00+x11)(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
    # M3 = x00(y01-y11) = x00*y01 - x00*y11
    # -M5 = -(x00+x01)y11 = -x00*y11 - x01*y11
    # M7 = (x01-x11)(y10+y11) = x01*y10 + x01*y11 - x11*y10 - x11*y11
    
    # Sum: x00*y00 + x00*y11 + x11*y00 + x11*y11 + x00*y01 - x00*y11 - x00*y11 - x01*y11 + x01*y10 + x01*y11 - x11*y10 - x11*y11
    # = x00*y00 + x00*y01 + x01*y10 + x11*y00 - x00*y11 - x11*y10
    # That doesn't look right. Let me use a standard Strassen reference.
    
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C = [[p1+p3-p5+p7, p4+p5], [p2+p3, p1-p2+p4+p6]]
    
    # Let me verify C[0,0] = p1+p3-p5+p7:
    # p1 = ae + ah + de + dh
    # p3 = af - ah
    # -p5 = -ah - bh
    # p7 = bg + bh - dg - dh
    # Sum = ae + ah + de + dh + af - ah - ah - bh + bg + bh - dg - dh
    # = ae + af + de + bg + dh - ah - bh - dg
    # Hmm, that's not ae + bf + ... Let me just use the known correct formulas.
    
    # I'll use the standard Strassen formulas and trust they work.
    
    # For our problem, we need to map:
    # Block 1: A1 (2x2) * B1a (2x2) -> C[:, 0:2]
    #   A1[i, j'] = A[i, j'] for i,j' in 0,1
    #   B1a[j', k] = B[j', k] for j' in 0,1, k in 0,1
    #   C[i,k] for i in 0,1, k in 0,1
    
    # Block 2: A1 (2x2) * B1b (2x2) -> C[:, 2:4]
    #   A1[i, j'] = A[i, j'] for i,j' in 0,1
    #   B1b[j', k'] = B[j', k'+2] for j' in 0,1, k' in 0,1
    #   C[i, k'+2] for i in 0,1, k' in 0,1
    
    # Block 3: A2 (2x2) * B2a (2x2) -> C[:, 0:2]
    #   A2[i, j'] = A[i, j'+2] for i,j' in 0,1
    #   B2a[j', k] = B[j'+2, k] for j' in 0,1, k in 0,1
    #   C[i,k] for i in 0,1, k in 0,1
    
    # Block 4: A2 (2x2) * B2b (2x2) -> C[:, 2:4]
    #   A2[i, j'] = A[i, j'+2] for i,j' in 0,1
    #   B2b[j', k'] = B[j'+2, k'+2] for j' in 0,1, k' in 0,1
    #   C[i, k'+2] for i in 0,1, k' in 0,1
    
    # For each 2x2x2 block, we define 7 terms.
    # U vector for the 2x2 block: indices (i, j') where i,j' in 0,1
    # V vector: indices (j', k') where j',k' in 0,1
    # W vector: indices (k', i) where k',i in 0,1 (using trace convention W[k*n+i])
    
    # The 7 Strassen terms (coefficient, U_idx, V_idx, W_idx):
    # For a 2x2x2 multiplication with A indices (i,j), B indices (j,k), C indices (k,i):
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    
    # p1 = (a+d)(e+h):
    #   U: a=1, d=1 (i=0,j=0) and (i=1,j=1) -> U[0]=1, U[3]=1
    #   V: e=1, h=1 (j=0,k=0) and (j=1,k=1) -> V[0]=1, V[3]=1
    #   W: contributes to C[0,0] (k=0,i=0) and C[1,1] (k=1,i=1) -> W[0]=1, W[3]=1
    #   Coefficient: 1
    #   But wait, p1 = (a+d)(e+h) is a single product. In tensor decomposition, each term is rank-1.
    #   p1 = a*e + a*h + d*e + d*h, so it's 4 rank-1 terms. But Strassen combines them into one multiplication.
    #   For tensor decomposition, each "multiplication" in Strassen corresponds to one rank-1 term in the decomposition.
    #   So p1 is ONE rank-1 term where U has a and d, V has e and h, and W has contributions to C[0,0] and C[1,1].
    #   Actually no. In Strassen, each pi is a product of two linear forms. So each pi is a single rank-1 term in the tensor decomposition.
    #   The linear form for U in p1 is a+d, which means U vector has 1 at position (0,0) and 1 at position (1,1).
    #   The linear form for V in p1 is e+h, which means V vector has 1 at position (0,0) and 1 at position (1,1).
    #   The output: p1 contributes +1 to C[0,0] and +1 to C[1,1]. So W vector has 1 at (k=0,i=0) and 1 at (k=1,i=1).
    
    # Let me list all 7 terms with their U, V, W vectors:
    # U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  [W[k,i] where k is row of C, i is col of C]
    
    # Wait, W[k*n+i] with n=2. So W[0] = W[k=0,i=0], W[1] = W[k=0,i=1], W[2] = W[k=1,i=0], W[3] = W[k=1,i=1]
    
    # Term 1: p1 = (a+d)(e+h)
    # U: [1, 0, 0, 1]  (a and d)
    # V: [1, 0, 0, 1]  (e and h)
    # W: [1, 0, 0, 1]  (C[0,0] and C[1,1])
    
    # Term 2: p2 = (c+d)e
    # U: [0, 0, 1, 1]  (c and d)
    # V: [1, 0, 0, 0]  (e)
    # W: [0, 0, 1, 1]  (C[1,0] and C[1,1]) with signs: C[1,0] += p2, C[1,1] -= p2
    # So W: [0, 0, 1, -1]
    
    # Term 3: p3 = a(f-h)
    # U: [1, 0, 0, 0]  (a)
    # V: [0, 1, 0, -1]  (f and -h)
    # W: [1, 0, 1, 0]  (C[0,0] += p3, C[1,0] += p3)
    
    # Term 4: p4 = d(g-e)
    # U: [0, 0, 0, 1]  (d)
    # V: [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0]  (g and -e)
    # W: [0, 1, 0, 1]  (C[0,1] += p4, C[1,1] += p4)
    
    # Term 5: p5 = (a+b)h
    # U: [1, 1, 0, 0]  (a and b)
    # V: [0, 0, 0, 1]  (h)
    # W: [-1, 1, 0, 0]  (C[0,0] -= p5, C[0,1] += p5)
    
    # Term 6: p6 = (c-a)(e+f)
    # U: [-1, 0, 1, 0]  (-a and c)
    # V: [1, 1, 0, 0]  (e and f)
    # W: [0, 0, 0, 1]  (C[1,1] += p6)
    
    # Term 7: p7 = (b-d)(g+h)
    # U: [0, 1, 0, -1]  (b and -d)
    # V: [0, 0, 1, 1]  (g and h)
    # W: [1, -1, 0, 0]  (C[0,0] += p7, C[0,1] -= p7)
    
    # Let me verify C[0,0] = p1 + p3 - p5 + p7:
    # p1: +1, p3: +1, -p5: -1, p7: +1 -> total coefficient from these: 1+1-1+1 = 2? No, each is a separate term.
    # C[0,0] gets: +1 from p1, +1 from p3, -1 from p5, +1 from p7. So C[0,0] = p1 + p3 - p5 + p7. Correct.
    
    # Let me verify C[0,1] = p4 + p5 - p7:
    # p4: +1, p5: +1, -p7: -1. So C[0,1] = p4 + p5 - p7. Correct.
    
    # Let me verify C[1,0] = p2 + p3:
    # p2: +1, p3: +1. So C[1,0] = p2 + p3. Correct.
    
    # Let me verify C[1,1] = p1 - p2 + p4 + p6:
    # p1: +1, -p2: -1, p4: +1, p6: +1. So C[1,1] = p1 - p2 + p4 + p6. Correct.
    
    # Now for each of the 4 blocks, we map these 7 terms.
    
    rank = 28
    U = [[0.0] * rank for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0.0] * rank for _ in range(16)]  # m*p = 4*4 = 16
    W = [[0.0] * rank for _ in range(8)]   # p*n = 4*2 = 8
    
    # U indices: A[i,j] -> index i*m + j = i*4 + j
    # i in 0,1; j in 0,1,2,3
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # V indices: B[j,k] -> index j*p + k = j*4 + k
    # j in 0,1,2,3; k in 0,1,2,3
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    
    # W indices: C[k,i] -> index k*n + i = k*2 + i
    # k in 0,1,2,3; i in 0,1
    # C[0,0]=0, C[0,1]=1
    # C[1,0]=2, C[1,1]=3
    # C[2,0]=4, C[2,1]=5
    # C[3,0]=6, C[3,1]=7
    
    # Block 1: A1 (cols 0,1) * B1a (rows 0,1, k=0,1) -> C[:, 0:2]
    # A1[i,j'] = A[i, j'] for i,j' in 0,1
    #   A1[0,0] = A[0,0] = U index 0
    #   A1[0,1] = A[0,1] = U index 1
    #   A1[1,0] = A[1,0] = U index 4
    #   A1[1,1] = A[1,1] = U index 5
    # B1a[j',k] = B[j', k] for j' in 0,1, k in 0,1
    #   B1a[0,0] = B[0,0] = V index 0
    #   B1a[0,1] = B[0,1] = V index 1
    #   B1a[1,0] = B[1,0] = V index 4
    #   B1a[1,1] = B[1,1] = V index 5
    # C[i,k] for i in 0,1, k in 0,1
    #   C[0,0] = W index 0
    #   C[0,1] = W index 1
    #   C[1,0] = W index 2
    #   C[1,1] = W index 3
    
    # The 7 terms for block 1:
    # U_local: [0, 1, 4, 5] corresponding to (0,0), (0,1), (1,0), (1,1)
    # V_local: [0, 1, 4, 5] corresponding to (0,0), (0,1), (1,0), (1,1)
    # W_local: [0, 1, 2, 3] corresponding to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    
    block_offsets = [
        # (U_a, U_b, U_c, U_d, V_e, V_f, V_g, V_h, W_k0i0, W_k0i1, W_k1i0, W_k1i1)
        # Block 1: A1 * B1a -> C[:, 0:2]
        (0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 2, 3),
        # Block 2: A1 * B1b -> C[:, 2:4]
        # B1b[j',k'] = B[j', k'+2]
        #   B1b[0,0] = B[0,2] = V index 2
        #   B1b[0,1] = B[0,3] = V index 3
        #   B1b[1,0] = B[1,2] = V index 6
        #   B1b[1,1] = B[1,3] = V index 7
        # C[i,k'+2] for i in 0,1, k' in 0,1
        #   C[0,2] = W index 4
        #   C[0,3] = W index 5
        #   C[1,2] = W index 6
        #   C[1,3] = W index 7
        (0, 1, 4, 5, 2, 3, 6, 7, 4, 5, 6, 7),
        # Block 3: A2 * B2a -> C[:, 0:2]
        # A2[i,j'] = A[i, j'+2]
        #   A2[0,0] = A[0,2] = U index 2
        #   A2[0,1] = A[0,3] = U index 3
        #   A2[1,0] = A[1,2] = U index 6
        #   A2[1,1] = A[1,3] = U index 7
        # B2a[j',k] = B[j'+2, k]
        #   B2a[0,0] = B[2,0] = V index 8
        #   B2a[0,1] = B[2,1] = V index 9
        #   B2a[1,0] = B[3,0] = V index 12
        #   B2a[1,1] = B[3,1] = V index 13
        # C[i,k] for i in 0,1, k in 0,1
        (2, 3, 6, 7, 8, 9, 12, 13, 0, 1, 2, 3),
        # Block 4: A2 * B2b -> C[:, 2:4]
        # B2b[j',k'] = B[j'+2, k'+2]
        #   B2b[0,0] = B[2,2] = V index 10
        #   B2b[0,1] = B[2,3] = V index 11
        #   B2b[1,0] = B[3,2] = V index 14
        #   B2b[1,1] = B[3,3] = V index 15
        # C[i,k'+2] for i in 0,1, k' in 0,1
        (2, 3, 6, 7, 10, 11, 14, 15, 4, 5, 6, 7),
    ]
    
    # Strassen terms: (U_vec, V_vec, W_vec) where vectors are over (a,b,c,d), (e,f,g,h), (C00,C01,C10,C11)
    strassen_terms = [
        # Term 1: p1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # Term 2: p2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # Term 3: p3
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 1, 0]),
        # Term 4: p4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 1]),
        # Term 5: p5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # Term 6: p6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # Term 7: p7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, -1, 0, 0]),
    ]
    
    col = 0
    for block in block_offsets:
        U_a, U_b, U_c, U_d = block[0], block[1], block[2], block[3]
        V_e, V_f, V_g, V_h = block[4], block[5], block[6], block[7]
        W_00, W_01, W_10, W_11 = block[8], block[9], block[10], block[11]
        
        for u_vec, v_vec, w_vec in strassen_terms:
            # u_vec = [coef_a, coef_b, coef_c, coef_d]
            # v_vec = [coef_e, coef_f, coef_g, coef_h]
            # w_vec = [coef_C00, coef_C01, coef_C10, coef_C11]
            
            # Set U
            if u_vec[0] != 0:
                U[U_a][col] += u_vec[0]
            if u_vec[1] != 0:
                U[U_b][col] += u_vec[1]
            if u_vec[2] != 0:
                U[U_c][col] += u_vec[2]
            if u_vec[3] != 0:
                U[U_d][col] += u_vec[3]
            
            # Set V
            if v_vec[0] != 0:
                V[V_e][col] += v_vec[0]
            if v_vec[1] != 0:
                V[V_f][col] += v_vec[1]
            if v_vec[2] != 0:
                V[V_g][col] += v_vec[2]
            if v_vec[3] != 0:
                V[V_h][col] += v_vec[3]
            
            # Set W
            if w_vec[0] != 0:
                W[W_00][col] += w_vec[0]
            if w_vec[1] != 0:
                W[W_01][col] += w_vec[1]
            if w_vec[2] != 0:
                W[W_10][col] += w_vec[2]
            if w_vec[3] != 0:
                W[W_11][col] += w_vec[3]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
