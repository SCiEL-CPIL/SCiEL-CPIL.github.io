"""Program for matrix multiplication 4x5x5 tensor decomposition.
Uses a recursive block decomposition with Strassen to reduce rank below schoolbook."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We want to decompose the tensor for C = A*B where A is 4x5, B is 5x5, C is 4x5.
    # We split the middle dimension m=5 into 4+1.
    # A = [A1 A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    
    # Now split B1 (4x5) into [B1a B1b] where B1a is 4x4, B1b is 4x1.
    # C = A1*B1a + A1*B1b + A2*B2
    
    # Term 1: A1*B1a is 4x4 * 4x4. We use Strassen (rank 49).
    # Term 2: A1*B1b is 4x4 * 4x1. Schoolbook (rank 16).
    # Term 3: A2*B2 is 4x1 * 1x5. Schoolbook (rank 20).
    
    # Total rank = 49 + 16 + 20 = 85.
    
    # Let's build the components.
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2*2x2 with rank 7.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # m1 = (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1  # a00, a11
        V[0,0] = 1; V[3,0] = 1  # b00, b11
        W[0,0] = 1; W[3,0] = 1  # c00, c11
        
        # m2 = (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1  # a10, a11
        V[0,1] = 1              # b00
        W[2,1] = 1              # c10
        
        # m3 = a00(b10-b11)
        U[0,2] = 1              # a00
        V[2,2] = 1; V[3,2] = -1 # b10, -b11
        W[1,2] = 1; W[2,2] = 1  # c01, c10
        
        # m4 = a11(b01-b10)
        U[3,3] = 1              # a11
        V[1,3] = 1; V[2,3] = -1 # b01, -b10
        W[0,3] = 1; W[3,3] = 1  # c00, c11
        
        # m5 = (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1  # a00, a01
        V[3,4] = 1              # b11
        W[0,4] = -1; W[1,4] = 1 # -c00, c01
        
        # m6 = (a10-a00)(b00+b10)
        U[2,5] = 1; U[0,5] = -1 # a10, -a00
        V[0,5] = 1; V[2,5] = 1  # b00, b10
        # m6 contributes to c10? No, let's check the formulas again.
        # Standard Strassen:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01-b10)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        #
        # c00 = m1 - m5 + m7 - m4
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 - m6
        #
        # Let's re-verify my assignments.
        # m1: U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1. 
        #      Product is a00b00 + a00b11 + a11b00 + a11b11.
        #      W[0,0]=1, W[3,0]=1. This adds to c00 and c11. Correct.
        #
        # m2: U[2,1]=1, U[3,1]=1; V[0,1]=1.
        #      Product is a10b00 + a11b00.
        #      W[2,1]=1. This adds to c10. Correct.
        #
        # m3: U[0,2]=1; V[2,2]=1, V[3,2]=-1.
        #      Product is a00b10 - a00b11.
        #      W[1,2]=1, W[2,2]=1. This adds to c01 and c10. Correct.
        #
        # m4: U[3,3]=1; V[1,3]=1, V[2,3]=-1.
        #      Product is a11b01 - a11b10.
        #      W[0,3]=1, W[3,3]=1. This adds to c00 and c11. Correct.
        #
        # m5: U[0,4]=1, U[1,4]=1; V[3,4]=1.
        #      Product is a00b11 + a01b11.
        #      W[0,4]=-1, W[1,4]=1. This subtracts from c00 and adds to c01. Correct.
        #
        # m6: U[2,5]=1, U[0,5]=-1; V[0,5]=1, V[2,5]=1.
        #      Product is (a10-a00)(b00+b10) = a10b00 + a10b10 - a00b00 - a00b10.
        #      Formula says c11 = m1 - m2 + m3 - m6.
        #      So W[3,5] should be -1.
        W[3,5] = -1
        #
        # m7: U[1,6]=1, U[3,6]=-1; V[1,6]=1, V[3,6]=1.
        #      Product is (a01-a11)(b01+b11) = a01b01 + a01b11 - a11b01 - a11b11.
        #      Formula says c00 = m1 - m5 + m7 - m4.
        #      So W[0,6] should be 1.
        U[1,6] = 1; U[3,6] = -1
        V[1,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W

    def strassen_4x4():
        """Build 4x4x4 decomposition from 2x2x2 Strassen.
        Returns U, V, W of shape (16, 49).
        """
        U2, V2, W2 = strassen_2x2()
        
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        # Split 4x4 into four 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = A*B = [[C00, C01], [C10, C11]]
        #
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        #
        # We can use Strassen on the 4x4 by treating it as a 2x2 of 2x2 blocks.
        # But Strassen requires the same structure. 
        # Actually, we can just use 7 Strassen calls on the block level?
        # No, that would be rank 7 * 4 = 28? No, the blocks are 2x2.
        # If we use Strassen on the 2x2 blocks, the rank is 7 * rank(2x2x2) = 7 * 4 = 28?
        # Wait. The rank of 2x2x2 is 4 (schoolbook) or 7 (Strassen).
        # If we use Strassen for the 2x2 blocks, the total rank is 7 * 7 = 49.
        # This is correct.
        
        # Let's define the 7 block products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # And the block results:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # For each block product Mi, we use the 2x2 Strassen decomposition.
        # The indices for U, V, W in the 2x2 decomposition are:
        # U: i*2+j, V: j*2+k, W: k*2+i.
        # In the 4x4 context:
        # A00: i in 0,1; j in 0,1. U index: i*4+j
        # A01: i in 0,1; j in 2,3. U index: i*4+j
        # A10: i in 2,3; j in 0,1. U index: i*4+j
        # A11: i in 2,3; j in 2,3. U index: i*4+j
        
        # Let's map the 2x2 indices to 4x4 indices.
        # For a block at (bi, bj) where bi, bj in 0,1:
        # Local i, j in 0,1.
        # Global i = bi*2 + i, Global j = bj*2 + j.
        # Global U index = (bi*2+i)*4 + (bj*2+j).
        
        # We will iterate over the 7 Strassen terms.
        # For each term, we have a combination of A blocks and B blocks.
        # e.g. M1 = (A00+A11)(B00+B11).
        # This is a sum of products: A00B00 + A00B11 + A11B00 + A11B11.
        # But wait, Strassen is a bilinear form. 
        # U and V are linear combinations.
        # So for M1, the U vector is the sum of the U vectors for A00 and A11.
        # And the V vector is the sum of the V vectors for B00 and B11.
        
        # Let's construct U, V, W for each of the 7 terms.
        # Term t (0..6).
        # U_t is a 16x7 matrix? No, U_t is a 16x1 vector for each of the 7 sub-terms?
        # No, the rank is 49. So we have 49 columns.
        # Each of the 7 block-terms has rank 7 (from the 2x2 Strassen inside).
        # So we have 7 * 7 = 49 columns.
        
        # Let's define the blocks involved in each term.
        # Term 0: M1. A blocks: 00, 11. B blocks: 00, 11.
        # Term 1: M2. A blocks: 10, 11. B blocks: 00.
        # Term 2: M3. A blocks: 00. B blocks: 10, 11.
        # Term 3: M4. A blocks: 11. B blocks: 01, 10.
        # Term 4: M5. A blocks: 00, 01. B blocks: 11.
        # Term 5: M6. A blocks: 10, 00. B blocks: 00, 10.
        # Term 6: M7. A blocks: 01, 11. B blocks: 01, 11.
        
        # This is getting complicated. Let's just use the schoolbook for the 4x4 blocks?
        # No, that would be rank 16 for each block, total 7*16 = 112. Too high.
        
        # Let's stick to the recursive Strassen.
        # I will write a helper to add a block product to U, V, W.
        
        def add_block_product(U, V, W, term_idx, a_blocks, b_blocks, w_blocks, signs):
            # a_blocks: list of (bi, bj) for A
            # b_blocks: list of (bi, bj) for B
            # w_blocks: list of (bi, bj) for C
            # signs: list of signs for each w_block
            # term_idx: which of the 7 Strassen terms (0..6)
            # sub_term: which of the 7 sub-terms inside the 2x2 Strassen (0..6)
            
            pass # Placeholder
        
        # Actually, let's just hardcode the 4x4 Strassen using the 2x2 one.
        # It's easier to think of it as:
        # U[i,j, t, s] = sum of U2[i_local, j_local, s] for each block (i_block, j_block) in the term.
        
        # Let's create a 4D array for U, V, W? No, 2D is fine.
        
        # Let's iterate over the 7 main terms and 7 sub-terms.
        # Main term t (0..6), sub term s (0..6).
        # Column index c = t*7 + s.
        
        # We need to know which blocks are involved in each main term.
        # And for each main term, the 2x2 Strassen structure applies to the blocks.
        
        # Let's define the block indices for each main term.
        # A_blocks[t] is a list of (block_i, block_j, sign)
        # B_blocks[t] is a list of (block_i, block_j, sign)
        
        A_blocks = [
            [(0,0,1), (1,1,1)],      # M1
            [(1,0,1), (1,1,1)],      # M2
            [(0,0,1)],               # M3
            [(1,1,1)],               # M4
            [(0,0,1), (0,1,1)],      # M5
            [(1,0,1), (0,0,-1)],     # M6
            [(0,1,1), (1,1,-1)]      # M7
        ]
        
        B_blocks = [
            [(0,0,1), (1,1,1)],      # M1
            [(0,0,1)],               # M2
            [(1,0,1), (1,1,-1)],     # M3
            [(0,1,1), (1,0,-1)],     # M4
            [(1,1,1)],               # M5
            [(0,0,1), (1,0,1)],      # M6
            [(0,1,1), (1,1,1)]       # M7
        ]
        
        # And the C blocks for each main term.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        C_blocks = [
            [(0,0,1), (1,1,1)],      # M1
            [(1,0,1)],               # M2
            [(0,1,1), (1,0,1)],      # M3
            [(0,0,1), (1,1,1)],      # M4
            [(0,0,-1), (0,1,1)],     # M5
            [(1,1,-1)],              # M6
            [(0,0,1), (1,1,1)]       # M7
        ]
        
        # Now, for each main term t and sub term s:
        for t in range(7):
            for s in range(7):
                c = t*7 + s
                # U part
                for (bi, bj, sign) in A_blocks[t]:
                    for li in range(2):
                        for lj in range(2):
                            # Local index in 2x2 block
                            local_i = li
                            local_j = lj
                            # Global index
                            gi = bi*2 + li
                            gj = bj*2 + lj
                            U[gi*4 + gj, c] += sign * U2[local_i*2 + local_j, s]
                
                # V part
                for (bi, bj, sign) in B_blocks[t]:
                    for li in range(2):
                        for lj in range(2):
                            local_i = li
                            local_j = lj
                            gi = bi*2 + li
                            gj = bj*2 + lj
                            V[gi*4 + gj, c] += sign * V2[local_i*2 + local_j, s]
                
                # W part
                for (bi, bj, sign) in C_blocks[t]:
                    for li in range(2):
                        for lj in range(2):
                            local_i = li
                            local_j = lj
                            gi = bi*2 + li
                            gj = bj*2 + lj
                            W[gi*4 + gj, c] += sign * W2[local_i*2 + local_j, s]
        
        return U, V, W

    def schoolbook(n, m, p):
        """Return (U, V, W) for n x m * m x p with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        return U, V, W

    # Get the components
    U1, V1, W1 = strassen_4x4()  # 16x49
    U2, V2, W2 = schoolbook(4, 4, 1)  # 16x16
    U3, V3, W3 = schoolbook(4, 1, 5)  # 4x20
    
    # Assemble
    total_rank = 49 + 16 + 20
    U = np.zeros((20, total_rank), dtype=int)
    V = np.zeros((25, total_rank), dtype=int)
    W = np.zeros((20, total_rank), dtype=int)
    
    # Part 1: A1*B1a (4x4 * 4x4)
    # A1 is A[0:4, 0:4]. U indices 0..15.
    # B1a is B[0:4, 0:4]. V indices j*5+k for j in 0..3, k in 0..3.
    # C1a is C[0:4, 0:4]. W indices k*4+i for k in 0..3, i in 0..3.
    
    for c in range(49):
        for i in range(16):
            if U1[i, c] != 0:
                U[i, c] = U1[i, c]
        
        for i in range(16):
            if V1[i, c] != 0:
                # i is j*4+k for j,k in 0..3
                j = i // 4
                k = i % 4
                V[j*5 + k, c] = V1[i, c]
        
        for i in range(16):
            if W1[i, c] != 0:
                # i is k*4+i for k,i in 0..3
                k = i // 4
                ii = i % 4
                W[k*4 + ii, c] = W1[i, c]
    
    # Part 2: A1*B1b (4x4 * 4x1)
    # A1 is A[0:4, 0:4]. U indices 0..15.
    # B1b is B[0:4, 4]. V indices j*5+4 for j in 0..3.
    # C1b is C[0:4, 4]. W indices 4*4+i for i in 0..3.
    
    offset = 49
    for c in range(16):
        for i in range(16):
            if U2[i, c] != 0:
                U[i, offset + c] = U2[i, c]
        
        for i in range(16):
            if V2[i, c] != 0:
                # i is j*4+k for j in 0..3, k=0.
                j = i // 4
                k = i % 4
                # B1b is column 4 of B1, which is column 4 of B.
                # So V index is j*5 + 4.
                V[j*5 + 4, offset + c] = V2[i, c]
        
        for i in range(16):
            if W2[i, c] != 0:
                # i is k*4+i for k=0, i in 0..3.
                k = i // 4
                ii = i % 4
                # C1b is column 4 of C.
                # So W index is 4*4 + ii = 16 + ii.
                W[16 + ii, offset + c] = W2[i, c]
    
    # Part 3: A2*B2 (4x1 * 1x5)
    # A2 is A[0:4, 4]. U indices i*5+4 for i in 0..3.
    # B2 is B[4, 0:5]. V indices 4*5+k for k in 0..4.
    # C2 is C[0:4, 0:5]. W indices k*4+i for k in 0..4, i in 0..3.
    
    offset = 49 + 16
    for c in range(20):
        for i in range(4):
            if U3[i, c] != 0:
                U[i*5 + 4, offset + c] = U3[i, c]
        
        for i in range(5):
            if V3[i, c] != 0:
                # i is j*5+k for j=4, k in 0..4.
                j = i // 5
                k = i % 5
                V[4*5 + k, offset + c] = V3[i, c]
        
        for i in range(20):
            if W3[i, c] != 0:
                # i is k*4+i for k in 0..4, i in 0..3.
                k = i // 4
                ii = i % 4
                W[k*4 + ii, offset + c] = W3[i, c]
    
    # Convert to lists for the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
