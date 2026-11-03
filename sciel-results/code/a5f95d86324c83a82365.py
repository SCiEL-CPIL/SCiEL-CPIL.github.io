"""
Tensor decomposition for 4x4 * 4x7 matrix multiplication.
Uses Strassen's algorithm for the 4x4x4 sub-problem (rank 49)
and schoolbook for the 4x4x3 sub-problem (rank 48).
Total rank: 97.
"""

def construct():
    n, m, p = 4, 4, 7
    
    # We'll use Strassen for the 4x4x4 part and schoolbook for 4x4x3
    # Strassen decomposes 4x4*4x4 into rank 49
    # Schoolbook for 4x4*4x3 is rank 48
    # Total rank = 49 + 48 = 97
    
    rank_total = 49 + 48  # 97
    
    U = [[0.0] * rank_total for _ in range(n * m)]   # 16 x 97
    V = [[0.0] * rank_total for _ in range(m * p)]   # 28 x 97
    W = [[0.0] * rank_total for _ in range(p * n)]   # 28 x 97
    
    # --- Strassen part: 4x4 * 4x4 ---
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 4x4x4, we can use the standard embedding.
    # 
    # The 4x4 matrix is partitioned into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen uses 7 products of 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each 2x2 product has rank 4 (schoolbook), so 7*4 = 28? No...
    # Actually each M_i is a 2x2 * 2x2 product, which itself can be decomposed.
    # But Strassen's rank-7 decomposition for 2x2x2 is the key.
    #
    # Let me think differently. The tensor for 4x4x4 matrix mult can be 
    # decomposed using the Kronecker product structure.
    #
    # Actually, let me use a direct approach. The 4x4x4 matrix multiplication
    # tensor has a known decomposition. Let me use the fact that:
    # rank(4x4x4) <= 49 via Strassen-style decomposition.
    #
    # Standard Strassen for 2x2x2: rank 7
    # For 4x4x4: we can view it as 2x2 blocks, each block multiplication
    # uses 7 rank-7 decompositions... but that gives 7*7=49 if we use
    # Strassen recursively, or 7*4=28 if we use schoolbook for inner blocks.
    #
    # Wait, let me be more careful. 
    #
    # The 4x4x4 tensor: we have indices i,j in {0,1,2,3} for A and B, 
    # and k for the sum. 
    #
    # Partition into 2x2 blocks:
    # A = [[A_00, A_01], [A_10, A_11]] where each A_ij is 2x2
    # B = [[B_00, B_01], [B_10, B_11]] where each B_ij is 2x2
    #
    # C_00 = A_00*B_00 + A_01*B_10
    # C_01 = A_00*B_01 + A_01*B_11
    # C_10 = A_10*B_00 + A_11*B_10
    # C_11 = A_10*B_01 + A_11*B_11
    #
    # Using Strassen for the 2x2 block-level:
    # M1 = (A_00+A_11)(B_00+B_11)
    # M2 = (A_10+A_11)B_00
    # M3 = A_00(B_01-B_11)
    # M4 = A_11(B_10-B_00)
    # M5 = (A_00+A_01)B_11
    # M6 = (A_10-A_00)(B_00+B_01)
    # M7 = (A_01-A_11)(B_10+B_11)
    #
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a product of two 2x2 matrices. If we decompose each 2x2x2
    # product using schoolbook (rank 4), then total rank = 7*4 = 28.
    # But wait, that's less than 49. Let me reconsider.
    #
    # Hmm, actually the issue is that M_i involves sums/differences of blocks,
    # so the decomposition isn't simply 7 independent 2x2x2 products.
    #
    # Let me use a different, more direct approach. I'll construct the
    # decomposition explicitly using the known rank-49 decomposition for 4x4x4.
    #
    # Actually, let me just use the schoolbook for everything but verify 
    # that Strassen gives us a saving.
    #
    # For 4x4x4: schoolbook rank = 64. Strassen gives rank 49.
    # For 4x4x3: schoolbook rank = 48.
    # Total with Strassen: 49 + 48 = 97.
    # Total schoolbook: 64 + 48 = 112.
    #
    # Let me construct the Strassen decomposition for 4x4x4 explicitly.
    #
    # I'll index the 4x4 matrices as:
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3
    # C[i,k] for i,k in 0..3
    #
    # Partition into 2x2 blocks:
    # Block (a,b) of A covers rows 2a..2a+1, cols 2b..2b+1
    # Block (a,b) of B covers rows 2a..2a+1, cols 2b..2b+1
    # Block (a,b) of C covers rows 2a..2a+1, cols 2b..2b+1
    #
    # Strassen's 7 combinations at the block level:
    # For each combination, we multiply two 2x2 matrices.
    # Each 2x2x2 multiplication uses schoolbook: 4 terms.
    # So each Strassen combination uses 4 rank-1 tensors.
    # Total: 7 * 4 = 28 rank-1 tensors for the 4x4x4 part.
    
    # Wait, that gives 28, not 49. Let me reconsider.
    
    # I think I'm confusing myself. Let me just directly verify:
    # The rank of the 4x4x4 matrix multiplication tensor is at most 49
    # (this is a known result). Let me construct it.
    
    # Actually, let me take a step back and just use a simpler approach:
    # Use schoolbook for the entire 4x4x7 problem, which gives rank 112.
    # Then try to find a better decomposition.
    
    # For now, let me implement the Strassen-based approach properly.
    
    # The key insight: 4x4x4 matrix mult can be done with 49 multiplications
    # using Strassen's algorithm applied to 2x2 blocks, where each 2x2x2
    # sub-problem is solved with rank 7 (Strassen for 2x2x2).
    
    # So: 7 (Strassen combos) * 7 (Strassen for each 2x2x2) = 49.
    
    # Let me construct this.
    
    # First, let me define the Strassen decomposition for 2x2x2.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # The 7 rank-1 tensors for 2x2x2:
    # Each has entries in U (2x2), V (2x2), W (2x2)
    
    # M1: U = [[1,0],[0,1]] * [[1,0],[1,1]]... no, let me think in terms
    # of the tensor decomposition.
    
    # For M1 = (a+d)(e+h):
    # U factor: a has coeff 1, d has coeff 1 -> U = [[1,0],[0,1]]
    # V factor: e has coeff 1, h has coeff 1 -> V = [[1,0],[0,1]]
    # W factor: contributes to i and l with coeff 1 -> W = [[1,0],[0,1]]
    
    # For M2 = (c+d)e:
    # U: c=1, d=1 -> U = [[0,0],[1,1]]
    # V: e=1 -> V = [[1,0],[0,0]]
    # W: contributes to k with coeff 1 -> W = [[0,0],[1,0]]
    
    # For M3 = a(f-h):
    # U: a=1 -> U = [[1,0],[0,0]]
    # V: f=1, h=-1 -> V = [[0,1],[0,-1]]
    # W: contributes to j with coeff 1 -> W = [[0,1],[0,0]]
    
    # For M4 = d(g-e):
    # U: d=1 -> U = [[0,0],[0,1]]
    # V: g=-1, e=1... wait, d(g-e) = d*g - d*e
    # V: g has coeff 1, e has coeff -1 -> V = [[-1,0],[1,0]]
    # W: contributes to i with coeff -1, k with coeff 1 -> W = [[-1,0],[1,0]]
    
    # For M5 = (a+b)h:
    # U: a=1, b=1 -> U = [[1,1],[0,0]]
    # V: h=1 -> V = [[0,0],[0,1]]
    # W: contributes to i with coeff -1, j with coeff 1 -> W = [[-1,1],[0,0]]
    
    # For M6 = (c-a)(e+f):
    # U: c=1, a=-1 -> U = [[-1,0],[1,0]]
    # V: e=1, f=1 -> V = [[1,1],[0,0]]
    # W: contributes to l with coeff 1 -> W = [[0,0],[0,1]]
    
    # For M7 = (b-d)(g+h):
    # U: b=1, d=-1 -> U = [[0,1],[0,-1]]
    # V: g=1, h=1 -> V = [[0,0],[1,1]]
    # W: contributes to i with coeff 1 -> W = [[1,0],[0,0]]
    
    # Let me verify: C = A*B
    # C[0,0] = a*e + b*g = M1 + M4 - M5 + M7
    #   = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    #   = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
    #   = ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + (-bh+bh) + bg
    #   = ae + bg ✓
    
    # C[0,1] = a*f + b*h = M3 + M5
    #   = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh ✓
    
    # C[1,0] = c*e + d*g = M2 + M4
    #   = (c+d)e + d(g-e) = ce+de+dg-de = ce+dg ✓
    
    # C[1,1] = c*f + d*h = M1 - M2 + M3 + M6
    #   = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    #   = ae+ah+de+dh - ce-de + af-ah + ce+cf-ae-af
    #   = (ae-ae) + (ah-ah) + (de-de) + (-ce+ce) + (af-af) + dh + cf
    #   = cf + dh ✓
    
    # Great! So the 7 rank-1 tensors for 2x2x2 are correct.
    
    # Now for 4x4x4, I'll use the block-level Strassen decomposition.
    # Each of the 7 block-level combinations involves multiplying two 2x2 matrices.
    # Each 2x2x2 multiplication is decomposed into 7 rank-1 tensors.
    # So total: 7 * 7 = 49 rank-1 tensors.
    
    # Let me define the 7 Strassen tensors for 2x2x2:
    # Each is (U_2x2, V_2x2, W_2x2)
    
    strassen_2x2 = [
        # M1
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # M2
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        # M3
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
        # M4
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[-1,0],[1,0]]),
        # M5
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
        # M6
        (([-1,0],[1,0]), [[1,1],[0,0]], [[0,0],[0,1]]),
        # M7
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Block-level Strassen combinations for 4x4x4:
    # Each combination is (U_blocks, V_blocks, W_blocks)
    # where U_blocks is a 2x2 grid of 2x2 blocks (i.e., 4x4),
    # but actually it's a linear combination of blocks.
    
    # Let me define the block-level coefficients.
    # For each of the 7 Strassen combinations, I need to specify:
    # - Which A-blocks are combined and with what coefficients
    # - Which B-blocks are combined and with what coefficients
    # - Which C-blocks are affected and with what coefficients
    
    # M1 = (A_00+A_11)(B_00+B_11)
    # M2 = (A_10+A_11)B_00
    # M3 = A_00(B_01-B_11)
    # M4 = A_11(B_10-B_00)
    # M5 = (A_00+A_01)B_11
    # M6 = (A_10-A_00)(B_00+B_01)
    # M7 = (A_01-A_11)(B_10+B_11)
    
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    # For each combination, the U factor is a 4x4 matrix formed by placing
    # the A-block coefficients, and similarly for V and W.
    
    # Let me define a helper to create a 4x4 matrix from 2x2 block coefficients.
    def make_4x4_from_blocks(block_coeffs):
        """block_coeffs is a 2x2 list of coefficients (or 2x2 matrices)."""
        result = [[0.0]*4 for _ in range(4)]
        for a in range(2):
            for b in range(2):
                coeff = block_coeffs[a][b]
                if isinstance(coeff, (int, float)):
                    for i in range(2):
                        for j in range(2):
                            result[2*a+i][2*b+j] = coeff
                else:
                    # coeff is a 2x2 matrix
                    for i in range(2):
                        for j in range(2):
                            result[2*a+i][2*b+j] = coeff[i][j]
        return result
    
    # Now, for each of the 7 block-level combinations, I need to form the
    # U, V, W factors for the 4x4x4 tensor.
    
    # The structure is:
    # For combination c (0..6), the U factor is a 4x4 matrix U_c,
    # the V factor is a 4x4 matrix V_c, and the W factor is a 4x4 matrix W_c.
    # But actually, since each block-level combination involves multiplying
    # two 2x2 matrices, and each such multiplication has 7 rank-1 terms,
    # I need to combine them.
    
    # Let me think about this more carefully.
    
    # For block-level combination M1 = (A_00+A_11)(B_00+B_11):
    # This is a product of two 2x2 matrices. Let's call them X and Y.
    # X = A_00 + A_11, Y = B_00 + B_11.
    # X*Y is decomposed into 7 rank-1 tensors using Strassen.
    # Each rank-1 tensor has U (2x2), V (2x2), W (2x2) factors.
    #
    # To embed this into the 4x4x4 tensor:
    # The U factor for the 4x4x4 tensor is a 4x4 matrix where:
    # - Block (0,0) gets the U factor from the 2x2x2 decomposition
    # - Block (1,1) gets the U factor from the 2x2x2 decomposition
    # - Other blocks are zero
    #
    # Similarly for V and W.
    
    # Let me define the block-level structure for each of the 7 combinations.
    
    # For each combination, I'll specify:
    # - A_blocks: dict mapping (a,b) -> coefficient for A_{a,b}
    # - B_blocks: dict mapping (a,b) -> coefficient for B_{a,b}
    # - C_blocks: dict mapping (a,b) -> coefficient for C_{a,b}
    
    block_combinations = [
        # M1 = (A_00+A_11)(B_00+B_11)
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1},  # W factor: contributes to C_00 and C_11
        },
        # M2 = (A_10+A_11)B_00
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1},  # W factor: contributes to C_10
        },
        # M3 = A_00(B_01-B_11)
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1},  # W factor: contributes to C_01
        },
        # M4 = A_11(B_10-B_00)
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): -1, (1,0): 1},  # W factor: contributes to C_00 (-1) and C_10 (1)
        },
        # M5 = (A_00+A_01)B_11
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1},  # W factor: contributes to C_00 (-1) and C_01 (1)
        },
        # M6 = (A_10-A_00)(B_00+B_01)
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1},  # W factor: contributes to C_11
        },
        # M7 = (A_01-A_11)(B_10+B_11)
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1},  # W factor: contributes to C_00
        },
    ]
    
    # Now, for each block-level combination and each of the 7 Strassen 2x2x2 tensors,
    # I'll create a rank-1 tensor for the 4x4x4 decomposition.
    
    col = 0
    for comb_idx, comb in enumerate(block_combinations):
        A_blocks = comb['A']
        B_blocks = comb['B']
        C_blocks = comb['C']
        
        for s_idx, (U2, V2, W2) in enumerate(strassen_2x2):
            # Create 4x4 U, V, W matrices
            
            U4 = [[0.0]*4 for _ in range(4)]
            V4 = [[0.0]*4 for _ in range(4)]
            W4 = [[0.0]*4 for _ in range(4)]
            
            # Fill U4 based on A_blocks
            for (a, b), coeff in A_blocks.items():
                for i in range(2):
                    for j in range(2):
                        U4[2*a+i][2*b+j] = coeff * U2[i][j]
            
            # Fill V4 based on B_blocks
            for (a, b), coeff in B_blocks.items():
                for i in range(2):
                    for j in range(2):
                        V4[2*a+i][2*b+j] = coeff * V2[i][j]
            
            # Fill W4 based on C_blocks
            for (a, b), coeff in C_blocks.items():
                for i in range(2):
                    for j in range(2):
                        W4[2*a+i][2*b+j] = coeff * W2[i][j]
            
            # Add to U, V, W
            for i in range(16):
                U[i][col] = U4[i//4][i%4] if False else U4[i//4][i%4]
            
            # Actually, U is 16x97, where row index = i*4+j for A[i,j]
            # Wait, U has shape (n*m, rank) = (16, rank). Row index is i*m+j = i*4+j.
            # So U[i*4+j, col] = U4[i][j]
            
            for i in range(4):
                for j in range(4):
                    U[i*4+j][col] = U4[i][j]
            
            # V has shape (m*p, rank) = (28, rank). Row index is j*p+k = j*7+k.
            # But V4 is 4x4, corresponding to B[j,k] for j,k in 0..3.
            # So V[j*7+k, col] = V4[j][k]
            
            for j in range(4):
                for k in range(4):
                    V[j*7+k][col] = V4[j][k]
            
            # W has shape (p*n, rank) = (28, rank). Row index is k*n+i = k*4+i.
            # W4 is 4x4, corresponding to C[i,k] for i,k in 0..3.
            # So W[k*4+i, col] = W4[i][k]
            
            for i in range(4):
                for k in range(4):
                    W[k*4+i][col] = W4[i][k]
            
            col += 1
    
    # Now add the schoolbook part for 4x4x3 (columns 4..6 of B)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j][col] = 1.0
                V[j*7+k][col] = 1.0
                W[k*4+i][col] = 1.0
                col += 1
    
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    return {"U": U, "V": V, "W": W}
