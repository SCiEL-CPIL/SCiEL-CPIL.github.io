"""
Matrix multiplication 4x4 x 4x9 tensor decomposition.
Uses two-level Strassen for 4x4x4 blocks (rank 49 each) and schoolbook for 4x4x1.
Total rank: 49 + 49 + 16 = 114
"""

import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll build the decomposition in three parts:
    # Part 1: 4x4 x 4x4 (B columns 0-3) using two-level Strassen: rank 49
    # Part 2: 4x4 x 4x4 (B columns 4-7) using two-level Strassen: rank 49
    # Part 3: 4x4 x 4x1 (B column 8) using schoolbook: rank 16
    # Total rank: 49 + 49 + 16 = 114
    
    total_rank = 114
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 114
    
    def add_term(rank_col, u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (idx, val), etc."""
        for idx, val in u_entries:
            U[idx][rank_col] += val
        for idx, val in v_entries:
            V[idx][rank_col] += val
        for idx, val in w_entries:
            W[idx][rank_col] += val
    
    def add_strassen_2x2_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, rank_offset):
        """
        Add Strassen 2x2x2 decomposition.
        A block: rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        B block: rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        C block: rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1
        """
        # Flatten indices: A[i,j] -> i*m + j, B[j,k] -> j*p + k, C[k,i] -> k*n + i
        # For 2x2 block starting at (r,c) in A:
        # A00 = A[r, c], A01 = A[r, c+1], A10 = A[r+1, c], A11 = A[r+1, c+1]
        # U indices: A00=r*m+c, A01=r*m+c+1, A10=(r+1)*m+c, A11=(r+1)*m+c+1
        
        a00 = a_row_offset * m + a_col_offset
        a01 = a_row_offset * m + a_col_offset + 1
        a10 = (a_row_offset + 1) * m + a_col_offset
        a11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        b00 = b_row_offset * p + b_col_offset
        b01 = b_row_offset * p + b_col_offset + 1
        b10 = (b_row_offset + 1) * p + b_col_offset
        b11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        c00 = c_row_offset * n + c_col_offset
        c01 = c_row_offset * n + c_col_offset + 1
        c10 = (c_row_offset + 1) * n + c_col_offset
        c11 = (c_row_offset + 1) * n + c_col_offset + 1
        
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        add_term(rank_offset + 0,
                 [(a00, 1), (a11, 1)],
                 [(b00, 1), (b11, 1)],
                 [(c00, 1), (c11, 1)])
        
        # M2 = (A10+A11)B00
        add_term(rank_offset + 1,
                 [(a10, 1), (a11, 1)],
                 [(b00, 1)],
                 [(c10, 1)])
        
        # M3 = A00(B01-B11)
        add_term(rank_offset + 2,
                 [(a00, 1)],
                 [(b01, 1), (b11, -1)],
                 [(c01, 1)])
        
        # M4 = A11(B10-B00)
        add_term(rank_offset + 3,
                 [(a11, 1)],
                 [(b10, 1), (b00, -1)],
                 [(c10, 1)])
        
        # M5 = (A00+A01)B11
        add_term(rank_offset + 4,
                 [(a00, 1), (a01, 1)],
                 [(b11, 1)],
                 [(c01, 1)])
        
        # M6 = (A10-A00)(B00+B01)
        add_term(rank_offset + 5,
                 [(a10, 1), (a00, -1)],
                 [(b00, 1), (b01, 1)],
                 [(c11, 1)])
        
        # M7 = (A01-A11)(B10+B11)
        add_term(rank_offset + 6,
                 [(a01, 1), (a11, -1)],
                 [(b10, 1), (b11, 1)],
                 [(c00, 1)])
    
    def add_strassen_4x4_block(b_col_base, rank_offset):
        """
        Add two-level Strassen for 4x4 x 4x4 block.
        B columns start at b_col_base.
        Uses 49 rank-1 terms.
        """
        # Split A into 2x2 blocks of 2x2:
        # A00: rows 0-1, cols 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        # Same for B (rows 0-3, cols b_col_base..b_col_base+3)
        # C: rows 0-3, cols b_col_base..b_col_base+3
        
        # 7 block-level Strassen products, each is 2x2 x 2x2
        # Each 2x2x2 uses 7 scalar Strassen terms
        # Total: 7 * 7 = 49
        
        # Block-level Strassen:
        # P1 = (A00+A11) * (B00+B11)
        # P2 = (A10+A11) * B00
        # P3 = A00 * (B01-B11)
        # P4 = A11 * (B10-B00)
        # P5 = (A00+A01) * B11
        # P6 = (A10-A00) * (B00+B01)
        # P7 = (A01-A11) * (B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Each Pi is a 2x2 matrix product. For each Pi, we need to do Strassen 2x2x2.
        # But the inputs to Pi are sums/differences of 2x2 blocks.
        
        # Let me think of this differently. The 4x4x4 tensor can be decomposed as
        # 49 rank-1 terms by applying Strassen recursively.
        
        # Alternative: directly enumerate the 49 terms.
        # At the top level, we have 7 "virtual" products, each of which is a 2x2x2 
        # multiplication with specific linear combinations.
        
        # Let me define the 7 block-level products and for each, do Strassen 2x2x2.
        
        # For product Pi = X * Y where X and Y are 2x2 matrices (which are linear combos 
        # of the 2x2 blocks of A and B), we do Strassen on X*Y.
        
        # Let me just directly construct all 49 terms.
        
        # Top-level Strassen indices for 4x4:
        # A blocks: A00(r0-1,c0-1), A01(r0-1,c2-3), A10(r2-3,c0-1), A11(r2-3,c2-3)
        # B blocks: B00(r0-1,c0-3), B01(r0-1,c4-7) [but we're in a 4-col block]
        # Actually B is 4x4 here: B00(r0-1,c0-1), B01(r0-1,c2-3), B10(r2-3,c0-1), B11(r2-3,c2-3)
        
        # Let me use a cleaner approach. I'll define helper to add a 2x2x2 Strassen
        # where the A and B inputs can be arbitrary linear combinations.
        
        # Actually, the cleanest way: the 4x4x4 multiplication tensor has rank 49.
        # I can construct it by taking the Kronecker product structure of two 2x2x2 Strassen.
        
        # Let me just hardcode the 49 terms by doing two levels explicitly.
        
        # Level 1: 7 block products
        # For each block product, the A-side is a sum of 2x2 blocks and B-side is a sum of 2x2 blocks
        # Each such product (sum of 2x2s) * (sum of 2x2s) needs to be expanded.
        
        # This is getting complex. Let me use a different approach:
        # Build the 4x4x4 decomposition by iterating over all 7x7=49 combinations
        # of the two Strassen levels.
        
        # Actually, let me just verify with a simpler method: use the known fact that
        # rank(4x4x4) <= 49 via Strassen, and construct it properly.
        
        # I'll define the 7 Strassen 2x2 terms as vectors, then take their tensor product.
        
        # Strassen 2x2 terms (U, V, W) as sparse dicts:
        s2_terms = [
            # (U_dict, V_dict, W_dict)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # M2
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),  # M3
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1}),  # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),  # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # M7
        ]
        
        # For 4x4, we apply this at two levels.
        # The 4x4 matrix A has indices (i,j) where i,j in 0..3.
        # We can view i = i1*2 + i0, j = j1*2 + j0 where i1,i0,j1,j0 in 0..1.
        # Similarly for B: (j,k) -> j = j1*2+j0, k = k1*2+k0
        # And C: (i,k) -> i = i1*2+i0, k = k1*2+k0
        
        # The 4x4x4 tensor is the "tensor square" of the 2x2x2 tensor.
        # Each rank-1 term in 4x4x4 is a Kronecker product of two 2x2x2 rank-1 terms.
        
        # Term (a,b) where a,b in 0..6:
        # U_4x4[i,j] = U2_a[i1,j1] * U2_b[i0,j0]
        # V_4x4[j,k] = V2_a[j1,k1] * V2_b[j0,k0]  
        # W_4x4[i,k] = W2_a[i1,k1] * W2_b[i0,k0]
        
        # Wait, this isn't quite right. The Strassen algorithm for 4x4 works by:
        # Treating A as a 2x2 matrix of 2x2 blocks, and doing Strassen at that level.
        # The "elements" of the 2x2 matrix of blocks are 2x2 matrices.
        # When we do Strassen, we form sums like (A00+A11) which is a 2x2 matrix.
        # Then we multiply (A00+A11) * (B00+B11) which is a 2x2 * 2x2 product.
        # This 2x2*2x2 product is done with Strassen, giving 7 scalar multiplications.
        
        # So the 49 terms come from: for each of 7 block-level Strassen products,
        # we do 7 scalar Strassen multiplications.
        
        # Let me define this more carefully.
        
        # Block-level Strassen for 4x4:
        # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
        # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
        # C blocks: C00=C[0:2,0:2], C01=C[0:2,2:4], C10=C[2:4,0:2], C11=C[2:4,2:4]
        
        # 7 products:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Each Pi is a 2x2 matrix. To compute Pi = Xi * Yi where Xi and Yi are 2x2,
        # we use Strassen 2x2, giving 7 scalar products.
        
        # For Pi, the U-side entries are in the 2x2 block of A, V-side in B, W-side in C.
        
        # Let me define for each block product Pi, the linear combination of A-blocks
        # and B-blocks, then do Strassen 2x2 on the resulting 2x2x2.
        
        # This is getting very involved. Let me just directly construct the 49 terms
        # by iterating.
        
        # For each of the 7 block products, I need to add 7 scalar Strassen terms.
        # The scalar Strassen terms operate on the 2x2 elements within each block.
        
        # Let me define a function that, given the A-block combination, B-block combination,
        # and C-block combination (with signs), adds the 7 Strassen 2x2 terms.
        
        # Actually, I think the cleanest approach is to just enumerate all 49 terms
        # by doing the two-level Strassen explicitly.
        
        # Let me use a different strategy: build the 4x4x4 decomposition by 
        # composing the 2x2x2 Strassen decomposition with itself.
        
        # The 2x2x2 Strassen has 7 terms. The 4x4x4 via Strassen has 7*7=49 terms.
        # Term (a,b) for a,b in 0..6:
        # U_4[i,j] where i=i1*2+i0, j=j1*2+j0:
        #   U_4[i,j] = U2_a[i1,j1] * U2_b[i0,j0]
        # V_4[j,k] where j=j1*2+j0, k=k1*2+k0:
        #   V_4[j,k] = V2_a[j1,k1] * V2_b[j0,k0]
        # W_4[i,k] where i=i1*2+i0, k=k1*2+k0:
        #   W_4[i,k] = W2_a[i1,k1] * W2_b[i0,k0]
        
        # This should work because the Strassen algorithm is bilinear and the 
        # 4x4 multiplication is the "tensor product" of two 2x2 multiplications
        # in the appropriate sense.
        
        # Let me verify this is correct. The 2x2 Strassen terms are:
        # s2_terms[a] = (U2_a, V2_a, W2_a)
        
        # For 4x4, term (a,b):
        # U_4[i1*2+i0, j1*2+j0] = U2_a[i1,j1] * U2_b[i0,j0]
        # V_4[j1*2+j0, k1*2+k0] = V2_a[j1,k1] * V2_b[j0,k0]
        # W_4[i1*2+i0, k1*2+k0] = W2_a[i1,k1] * W2_b[i0,k0]
        
        # This is the standard way to get the 4x4x4 decomposition from 2x2x2.
        
        # Let me implement this.
        
        for a in range(7):
            for b in range(7):
                rank_col = rank_offset + a * 7 + b
                U2_a, V2_a, W2_a = s2_terms[a]
                U2_b, V2_b, W2_b = s2_terms[b]
                
                # Build U_4, V_4, W_4 for this term
                u_entries = []
                v_entries = []
                w_entries = []
                
                for (i1, j1), ua in U2_a.items():
                    for (i0, j0), ub in U2_b.items():
                        i = i1 * 2 + i0
                        j = j1 * 2 + j0
                        val = ua * ub
                        if val != 0:
                            u_idx = i * m + j  # m=4
                            u_entries.append((u_idx, val))
                
                for (j1, k1), va in V2_a.items():
                    for (j0, k0), vb in V2_b.items():
                        j = j1 * 2 + j0
                        k = k1 * 2 + k0
                        val = va * vb
                        if val != 0:
                            v_idx = j * p + k  # p=9, but we're in a 4-col block
                            # k is relative to b_col_base
                            v_idx = j * p + (b_col_base + k)
                            v_entries.append((v_idx, val))
                
                for (i1, k1), wa in W2_a.items():
                    for (i0, k0), wb in W2_b.items():
                        i = i1 * 2 + i0
                        k = k1 * 2 + k0
                        val = wa * wb
                        if val != 0:
                            w_idx = (b_col_base + k) * n + i  # p*n indexing: k*n + i
                            w_entries.append((w_idx, val))
                
                add_term(rank_col, u_entries, v_entries, w_entries)
    
    # Part 1: 4x4 x 4x4, B columns 0-3
    add_strassen_4x4_block(0, 0)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 2: 4x4 x 4x4, B columns 4-7
    add_strassen_4x4_block(4, 49)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Part 3: 4x4 x 4x1, B column 8 (schoolbook)
    # Rank 16: for each i in 0..3, j in 0..3: one term
    for i in range(4):
        for j in range(4):
            rank_col = 98 + i * 4 + j
            U[i * m + j][rank_col] = 1
            V[j * p + 8][rank_col] = 1
            W[8 * n + i][rank_col] = 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
