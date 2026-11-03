import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 4, 4, 5
    
    # We want to decompose M_{4,4,5}
    # Strategy: Use the known rank-49 decomposition for M_{4,4,4} (two-level Strassen) 
    # and add 16 rank-1 terms for the 5th column (schoolbook).
    # Total rank = 49 + 16 = 65.
    
    # The current seed program returns rank 80 (schoolbook).
    # We will implement the rank-65 decomposition.
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def A_idx(i, j):
        return i * m + j

    def B_idx(j, k):
        return j * p + k

    def C_idx(k, i):
        return k * n + i

    # --- Part 1: M_{4,4,4} using Rank-49 Decomposition ---
    # This uses Strassen's algorithm recursively.
    # Level 1: 4x4x4 = 7 * (2x2x2)
    # Level 2: Each 2x2x2 = 7 * (1x1x1)
    # Total terms: 7 * 7 = 49
    
    # Let's define the 7 Strassen steps for the top level.
    # A, B, C are 4x4 matrices.
    # Blocks are 2x2.
    
    # S1 = A00 + A11, T1 = B00 + B11 -> P1
    # S2 = A10 + A11, T2 = B00 -> P2
    # S3 = A00, T3 = B10 - B11 -> P3
    # S4 = A11, T4 = B01 - B00 -> P4
    # S5 = A00 + A01, T5 = B11 -> P5
    # S6 = A10 - A00, T6 = B00 + B01 -> P6
    # S7 = A01 - A11, T7 = B10 + B11 -> P7
    
    # C00 = P1 - P4 + P5 + P6
    # C01 = P3 + P4
    # C10 = P2 + P3
    # C11 = P1 - P2 + P5 + P7
    
    # For each Pi, which is a 2x2 matrix product, we use the 7-term Strassen decomposition.
    # Let's write a helper to add a 2x2x2 decomposition.
    
    def add_2x2_x_2x2(A_block_coeffs, B_block_coeffs, C_block_coeffs, col_start):
        """
        Adds the rank-1 terms for a 2x2x2 multiplication.
        A_block_coeffs: dict mapping (i,j) to coeff
        B_block_coeffs: dict mapping (j,k) to coeff
        C_block_coeffs: dict mapping (k,i) to coeff
        Note: The 2x2 blocks correspond to specific indices in the 4x4 matrices.
        However, since the blocks are disjoint in the 4x4 case, we can map them directly.
        
        Actually, it's easier to just hardcode the 49 terms by iterating.
        """
        pass # We won't use this generic helper, we'll do it explicitly or with a loop.

    # Let's explicitly generate the terms.
    # We will iterate over the 7 top-level products (P1..P7).
    # For each, we need the coefficients of the A entries, B entries, and C entries.
    
    # Top-level definitions:
    # P1: A: { (0,0):1, (0,1):0, (1,0):0, (1,1):1, (2,0):0, (2,1):0, (3,0):0, (3,1):1, (0,2):0, (0,3):0, (1,2):0, (1,3):0, (2,2):0, (2,3):0, (3,2):0, (3,3):1 }
    # Wait, A is 4x4. Indices (i,j) for i,j in 0..3.
    # A00 is rows 0,1 and cols 0,1.
    # A11 is rows 2,3 and cols 2,3.
    # So A11[0,0] is A[2,2], A11[0,1] is A[2,3], A11[1,0] is A[3,2], A11[1,1] is A[3,3].
    
    # Let's define the A and B coefficient vectors for each top-level product.
    # We need to be careful with the indices.
    
    # A indices: 0..15
    # B indices: 0..19 (since m=4, p=5, but for M_{4,4,4} part, we only use k=0..3, so B indices 0..15)
    # C indices: 0..19 (since p=5, n=4, so C indices 0..19. For M_{4,4,4} part, we use k=0..3, so C indices 0..15)
    
    # Let's list the A coefficients for each of the 7 products (size 16, only 4 non-zero for each 2x2 block)
    # Product 1: (A00+A11)
    A1 = {0:1, 1:0, 4:0, 5:1, 12:0, 13:0, 14:0, 15:1} # 0:(0,0), 1:(0,1), 4:(1,0), 5:(1,1), 12:(2,2), 13:(2,3), 14:(3,2), 15:(3,3)
    # Wait, A_idx(i,j) = i*4+j.
    # A00: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # A11: (2,2)=10? No. 2*4+2=10. 2*4+3=11. 3*4+2=14. 3*4+3=15.
    # So A11 indices: 10, 11, 14, 15.
    
    A1 = {0:1, 1:0, 4:0, 5:1, 10:0, 11:0, 14:0, 15:1}
    # Actually, let's just write a function to build the dict.
    
    def get_A_block(block_id):
        # block_id: 00, 01, 10, 11
        r0, r1 = [0,1] if block_id[0]=='0' else [2,3]
        c0, c1 = [0,1] if block_id[1]=='0' else [2,3]
        return [A_idx(r0,c0), A_idx(r0,c1), A_idx(r1,c0), A_idx(r1,c1)]
        
    def get_B_block(block_id):
        # block_id: 00, 01, 10, 11
        r0, r1 = [0,1] if block_id[0]=='0' else [2,3]
        c0, c1 = [0,1] if block_id[1]=='0' else [2,3]
        return [B_idx(r0,c0), B_idx(r0,c1), B_idx(r1,c0), B_idx(r1,c1)]
        
    def get_C_block(block_id):
        # block_id: 00, 01, 10, 11 (C is k*n+i, so k is row, i is col)
        r0, r1 = [0,1] if block_id[0]=='0' else [2,3]
        c0, c1 = [0,1] if block_id[1]=='0' else [2,3]
        return [C_idx(r0,c0), C_idx(r0,c1), C_idx(r1,c0), C_idx(r1,c1)]

    # Top-level products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # For each Pi, we have a 2x2 * 2x2 product.
    # Let's define the Strassen 2x2 decomposition again.
    # p1 = (x00+x11)(y00+y11)
    # p2 = (x10+x11)y00
    # p3 = x00(y10-y11)
    # p4 = x11(y01-y00)
    # p5 = (x00+x01)y11
    # p6 = (x10-x00)(y00+y01)
    # p7 = (x01-x11)(y10+y11)
    
    # z00 = p1 - p4 + p5 + p6
    # z01 = p3 + p4
    # z10 = p2 + p3
    # z11 = p1 - p2 + p5 + p7
    
    # We will iterate over the 7 top-level products.
    # For each, we determine the coefficients of x00,x01,x10,x11 and y00,y01,y10,y11.
    # Then we generate the 7 rank-1 terms for the 2x2 product.
    # Each of those terms corresponds to 1 A-index, 1 B-index, and 4 C-indices (with coeffs).
    
    col = 0
    
    # Define the 7 top-level products' A and B block coefficients.
    # A_block_coeffs: dict from block name to coeff
    # B_block_coeffs: dict from block name to coeff
    
    top_products = [
        {'A': {'00':1, '11':1}, 'B': {'00':1, '11':1}, 'C': {'00':1, '01':1, '10':1, '11':1}}, # P1 contributes to all
        {'A': {'10':1, '11':1}, 'B': {'00':1}, 'C': {'10':1, '11':-1}}, # P2
        {'A': {'00':1}, 'B': {'10':1, '11':-1}, 'C': {'01':1, '10':1}}, # P3
        {'A': {'11':1}, 'B': {'01':1, '00':-1}, 'C': {'00':-1, '01':1}}, # P4
        {'A': {'00':1, '01':1}, 'B': {'11':1}, 'C': {'00':1, '11':1}}, # P5
        {'A': {'10':1, '00':-1}, 'B': {'00':1, '01':1}, 'C': {'00':1, '10':1}}, # P6
        {'A': {'01':1, '11':-1}, 'B': {'10':1, '11':1}, 'C': {'01':-1, '11':1}}, # P7
    ]
    
    # For each top product, we need to expand the 2x2*2x2.
    # The "X" matrix (2x2) is formed by the A blocks.
    # The "Y" matrix (2x2) is formed by the B blocks.
    
    for prod in top_products:
        A_blocks = prod['A']
        B_blocks = prod['B']
        C_blocks = prod['C']
        
        # The 2x2 X matrix entries:
        # x00 = sum(A[00]*coeff)
        # x01 = sum(A[01]*coeff) -- wait, the blocks are 2x2.
        # The 2x2 product is (A_block * B_block).
        # A_block is a 2x2 matrix. B_block is a 2x2 matrix.
        # Let's say A_block has entries a00, a01, a10, a11.
        # B_block has entries b00, b01, b10, b11.
        
        # The 7 terms of Strassen 2x2 are:
        # p1 = (a00+a11)(b00+b11)
        # p2 = (a10+a11)b00
        # p3 = a00(b10-b11)
        # p4 = a11(b01-b00)
        # p5 = (a00+a01)b11
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        
        # We need to map a00, etc., to actual A indices.
        # And b00, etc., to actual B indices.
        
        # Let's build the list of (a_idx, b_idx, c_idxs, coeffs) for each of the 7 terms.
        
        # Helper to get indices for a block
        def get_block_indices(block_id, type):
            # type: 'A', 'B', 'C'
            r0, r1 = [0,1] if block_id[0]=='0' else [2,3]
            c0, c1 = [0,1] if block_id[1]=='0' else [2,3]
            if type == 'A':
                return [A_idx(r0,c0), A_idx(r0,c1), A_idx(r1,c0), A_idx(r1,c1)]
            elif type == 'B':
                return [B_idx(r0,c0), B_idx(r0,c1), B_idx(r1,c0), B_idx(r1,c1)]
            else: # C
                return [C_idx(r0,c0), C_idx(r0,c1), C_idx(r1,c0), C_idx(r1,c1)]

        # We will iterate over the 7 Strassen terms for this product.
        # Term 1: (a00+a11)(b00+b11)
        # This is a product of sums.
        # We expand it into 4 rank-1 terms:
        # a00*b00, a00*b11, a11*b00, a11*b11
        
        # We need to know which A blocks are involved.
        # The "a00" in the 2x2 context refers to the top-left element of the 2x2 block.
        # But the 2x2 block itself is a combination of A blocks.
        # This is where it gets tricky.
        
        # Let's re-interpret.
        # The top-level product Pi is a sum of terms like (Sum of A blocks) * (Sum of B blocks).
        # Let X = Sum_i alpha_i A_i, where A_i are 2x2 blocks.
        # Let Y = Sum_j beta_j B_j, where B_j are 2x2 blocks.
        # Z = X * Y = Sum_i,j alpha_i beta_j (A_i * B_j).
        # Each (A_i * B_j) is a 2x2 matrix product.
        # We can decompose each (A_i * B_j) into 7 rank-1 terms.
        # Total terms = 7 * (number of non-zero pairs (i,j)).
        
        # For P1: X = A00+A11, Y = B00+B11.
        # Pairs: (00,00), (00,11), (11,00), (11,11).
        # 4 pairs. Each has 7 terms. Total 28 terms?
        # But wait, Strassen is for ONE 2x2 product.
        # If X and Y are sums, we can distribute the multiplication.
        # Z = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # Each of these is a 2x2 * 2x2 product.
        # Each can be done in 7 terms.
        # So P1 costs 4 * 7 = 28 terms?
        # That would make the total rank much higher.
        
        # NO. Strassen's algorithm replaces the 8 multiplications of schoolbook with 7.
        # The 7 multiplications are of the ELEMENTS of the 2x2 matrices.
        # If the 2x2 matrices are themselves sums of blocks, we CANNOT just distribute Strassen.
        # We must first form the sums X and Y, then apply Strassen to X and Y.
        
        # So, for P1:
        # X = A00 + A11.
        # Y = B00 + B11.
        # We apply the 7-term Strassen decomposition to X and Y.
        # p1 = (x00+x11)(y00+y11)
        # x00 = A00[0,0] + A11[0,0]
        # x11 = A00[1,1] + A11[1,1]
        # y00 = B00[0,0] + B11[0,0]
        # y11 = B00[1,1] + B11[1,1]
        
        # p1 = (A00[0,0] + A11[0,0] + A00[1,1] + A11[1,1]) * (B00[0,0] + B11[0,0] + B00[1,1] + B11[1,1])
        
        # This is a product of two sums of 4 terms each.
        # It expands to 4 * 4 = 16 rank-1 terms.
        # Similarly for p2..p7.
        
        # Let's count the number of terms.
        # p1: 16
        # p2: (x10+x11)y00. x10 (2 terms), x11 (2 terms), y00 (2 terms). (2+2)*2 = 8 terms.
        # p3: x00(y10-y11). x00 (2), y10 (2), y11 (2). 2*2 + 2*2 = 8 terms.
        # p4: x11(y01-y00). x11 (2), y01 (2), y00 (2). 8 terms.
        # p5: (x00+x01)y11. (2+2)*2 = 8 terms.
        # p6: (x10-x00)(y00+y01). (2+2)*(2+2) = 16 terms.
        # p7: (x01-x11)(y10+y11). (2+2)*(2+2) = 16 terms.
        
        # Total for P1: 16+8+8+8+8+16+16 = 80 terms.
        # That's way more than 7!
        
        # I made a mistake in my understanding.
        # Strassen's algorithm works because the 7 products are of the ELEMENTS.
        # If the elements are themselves sums, the number of terms explodes.
        
        # HOWEVER, in the two-level Strassen for 4x4, we treat the 2x2 blocks as "elements".
        # We do NOT expand the blocks. We treat them as atomic units in the algebra.
        # But for a tensor decomposition, we need rank-1 terms over the scalar field.
        
        # The rank of the tensor M_{4,4,4} is 49.
        # This means there ARE 49 rank-1 terms.
        # How are they constructed?
        
        # They are constructed by taking the 7 top-level products.
        # Each top-level product is a 2x2 * 2x2.
        # Each 2x2 * 2x2 is decomposed into 7 rank-1 terms.
        # Total 49.
        
        # But wait, if I use Strassen on the 2x2 level, I am assuming the 2x2 matrices are over the same field.
        # The "elements" of the 2x2 matrices in the top-level are the BLOCKS.
        # If I decompose the block multiplication into rank-1 terms, I am effectively doing schoolbook on the blocks?
        
        # No. The rank-1 terms are of the form:
        # u_i * v_i * w_i
        # where u_i, v_i, w_i are vectors.
        
        # Let's look at the structure of the 49 terms.
        # They correspond to the 7*7 combinations.
        # Each combination is a product of:
        # (A_element) * (B_element) * (C_element)
        
        # Specifically, for the two-level Strassen:
        # The 49 terms are generated by:
        # For each of the 7 top-level products (P1..P7):
        #    For each of the 7 bottom-level Strassen terms (p1..p7):
        #       Generate 1 rank-1 term.
        
        # Let's verify the structure of one term.
        # Top-level P1 = (A00+A11)(B00+B11).
        # Bottom-level p1 = (x00+x11)(y00+y11).
        # This p1 contributes to z00 and z11.
        # x00 = A00[0,0] + A11[0,0]
        # x11 = A00[1,1] + A11[1,1]
        # y00 = B00[0,0] + B11[0,0]
        # y11 = B00[1,1] + B11[1,1]
        
        # The rank-1 term for this specific (P1, p1) combination is NOT a single scalar multiplication.
        # It is a sum of scalar multiplications?
        
        # No, a rank-1 term in the tensor decomposition is a single product u * v * w.
        # If u is a vector with multiple non-zero entries, it's still one term.
        
        # So, for (P1, p1), we have:
        # u = vector with non-zero entries at A00[0,0], A11[0,0], A00[1,1], A11[1,1]
        # v = vector with non-zero entries at B00[0,0], B11[0,0], B00[1,1], B11[1,1]
        # w = vector with non-zero entries at C00[0,0], C00[1,1] (since z00, z11)
        
        # This is ONE rank-1 term!
        # The vectors u, v, w can have multiple non-zero entries.
        
        # So, the 49 terms are:
        # For each (top_prod, bottom_prod):
        #    u = linear combination of A entries
        #    v = linear combination of B entries
        #    w = linear combination of C entries
        
        # We just need to construct these 49 vectors.
        
        # Let's implement this.
        
        # We'll use the structure derived above.
        
        # We need a function to get the A coefficients for a given top product and bottom term.
        
        # Let's define the A and B coefficients for each bottom term, given the top product's blocks.
        
        # Top product defines the mapping from block IDs to coefficients.
        # Bottom term defines the structure (e.g., p1 uses x00, x11, y00, y11).
        
        # We'll iterate through the 49 combinations.
        
        for t_idx, prod in enumerate(top_products):
            A_blocks = prod['A']
            B_blocks = prod['B']
            C_blocks = prod['C']
            
            # Bottom-level Strassen terms
            # p1: x00+x11, y00+y11 -> z00, z11
            # p2: x10+x11, y00 -> z10, z11
            # p3: x00, y10-y11 -> z10, z01
            # p4: x11, y01-y00 -> z01, z00
            # p5: x00+x01, y11 -> z00, z11
            # p6: x10-x00, y00+y01 -> z00, z10
            # p7: x01-x11, y10+y11 -> z11, z01
            
            # We'll define the coefficients for A and B for each p.
            # A_coeffs: list of (block_id, pos, coeff)
            # B_coeffs: list of (block_id, pos, coeff)
            # C_coeffs: list of (block_id, pos, coeff)
            
            p_terms = [
                # p1
                {
                    'A': [('00',0,1),('11',0,1),('00',3,1),('11',3,1)], # x00, x11. Pos 0 is (0,0), Pos 3 is (1,1)
                    'B': [('00',0,1),('11',0,1),('00',3,1),('11',3,1)],
                    'C': [('00',0,1),('11',0,1),('00',3,1),('11',3,1)] # z00, z11. Pos 0 (0,0), Pos 3 (1,1)
                },
                # p2
                {
                    'A': [('10',1,1),('11',1,1),('10',3,1),('11',3,1)], # x10, x11. Pos 1 (1,0), Pos 3 (1,1)
                    'B': [('00',0,1)], # y00
                    'C': [('10',0,1),('11',0,1),('10',3,1),('11',3,-1)] # z10 (+), z11 (-)
                },
                # p3
                {
                    'A': [('00',0,1)], # x00
                    'B': [('10',1,1),('11',1,1),('10',3,-1),('11',3,-1)], # y10-y11
                    'C': [('10',0,1),('01',0,1),('10',3,1),('01',3,1)] # z10, z01
                },
                # p4
                {
                    'A': [('11',3,1)], # x11
                    'B': [('01',2,1),('00',2,-1),('01',3,1),('00',3,-1)], # y01-y00
                    'C': [('01',0,1),('00',0,-1),('01',3,1),('00',3,-1)] # z01, z00
                },
                # p5
                {
                    'A': [('00',0,1),('01',0,1),('00',1,1),('01',1,1)], # x00+x01
                    'B': [('11',3,1)], # y11
                    'C': [('00',0,1),('11',0,1),('00',3,1),('11',3,1)] # z00, z11
                },
                # p6
                {
                    'A': [('10',1,1),('00',1,-1),('10',3,1),('00',3,-1)], # x10-x00
                    'B': [('00',0,1),('01',0,1),('00',2,1),('01',2,1)], # y00+y01
                    'C': [('00',0,1),('10',0,1),('00',3,1),('10',3,1)] # z00, z10
                },
                # p7
                {
                    'A': [('01',1,1),('11',1,-1),('01',3,1),('11',3,-1)], # x01-x11
                    'B': [('10',1,1),('11',1,1),('10',3,1),('11',3,1)], # y10+y11
                    'C': [('11',0,1),('01',0,-1),('11',3,1),('01',3,-1)] # z11, z01
                }
            ]
            
            for b_term in p_terms:
                # Construct U, V, W vectors for this term
                u_vec = [0]*16
                v_vec = [0]*20
                w_vec = [0]*20
                
                for (block_id, pos, coeff) in b_term['A']:
                    if block_id in A_blocks:
                        # Add coeff * A_blocks[block_id] to u_vec at the position
                        # We need to map the 2x2 block position to the 4x4 index
                        # pos 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                        r = pos // 2
                        c = pos % 2
                        # Block offset
                        br = 0 if block_id[0]=='0' else 2
                        bc = 0 if block_id[1]=='0' else 2
                        idx = A_idx(br+r, bc+c)
                        u_vec[idx] += coeff * A_blocks[block_id]
                        
                for (block_id, pos, coeff) in b_term['B']:
                    if block_id in B_blocks:
                        r = pos // 2
                        c = pos % 2
                        br = 0 if block_id[0]=='0' else 2
                        bc = 0 if block_id[1]=='0' else 2
                        idx = B_idx(br+r, bc+c)
                        v_vec[idx] += coeff * B_blocks[block_id]
                        
                for (block_id, pos, coeff) in b_term['C']:
                    if block_id in C_blocks:
                        r = pos // 2
                        c = pos % 2
                        br = 0 if block_id[0]=='0' else 2
                        bc = 0 if block_id[1]=='0' else 2
                        idx = C_idx(br+r, bc+c)
                        w_vec[idx] += coeff * C_blocks[block_id]
                        
                # Add to U, V, W
                for i in range(16):
                    if u_vec[i] != 0:
                        U[i][col] = u_vec[i]
                for i in range(20):
                    if v_vec[i] != 0:
                        V[i][col] = v_vec[i]
                for i in range(20):
                    if w_vec[i] != 0:
                        W[i][col] = w_vec[i]
                        
                col += 1
                
    # --- Part 2: The 5th column (k=4) ---
    # This is a 4x4 * 4x1 product.
    # Rank 16 (schoolbook).
    # A is 4x4, B is 4x1 (column vector), C is 1x4 (row vector).
    # C[k=4, i] = sum_j A[i,j] * B[j, 4]
    
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

