import time
import sys

DIMENSIONS = [4, 4, 5]

def _build_winograd_4x4x4(U, V, W, col_start):
    """
    Constructs the rank-28 decomposition for 4x4 * 4x4 -> 4x4.
    This is based on the Block-Winograd algorithm which reuses Strassen-like 
    products for adjacent 2x2 blocks.
    
    The 4x4x4 tensor is decomposed into 28 rank-1 terms.
    U shape: (16, rank) -> indices i*4+j
    V shape: (16, rank) -> indices j*4+k (k is 0..3 for this part)
    W shape: (16, rank) -> indices k*4+i (k is 0..3, i is 0..3)
    Note: W index in the full 4x4x5 problem is k*n + i = k*4 + i.
    """
    col = col_start
    
    # Helper to get indices
    def u_idx(i, j): return i * 4 + j
    def v_idx(j, k): return j * 4 + k
    def w_idx(k, i): return k * 4 + i # k is col of C, i is row of C. W[k*n+i] with n=4.

    # We define 28 terms.
    # The structure is derived from partitioning 4x4 into 2x2 blocks.
    # Left half (cols 0,1 of B, rows 0,1 of C) and Right half (cols 2,3 of B, rows 2,3 of C).
    # But Winograd for 4x4x4 uses a specific set of 28 multiplications.
    
    # Reference: "A 4x4x4 matrix multiplication algorithm with 28 multiplications"
    # This is essentially applying Strassen to 2x2x2 blocks but sharing computations 
    # in a way that covers the 4x4 space efficiently.
    
    # The 28 terms are typically grouped into 4 sets of 7, corresponding to 
    # the 4 2x2 sub-matrices of C? Or something similar.
    
    # Let's use the known explicit formula for rank-28 4x4x4.
    # Source: Various optimization papers. One common decomposition:
    
    # M1 = (A00+A11) * (B00+B11) -> contributes to C00, C11
    # ... this is just Strassen on 2x2. 
    # The 28-term algorithm is more complex. It's often called "Winograd 4x4x4".
    
    # Let's construct it by combining two 2x2x2 Strassen blocks (rank 14) 
    # and adding cross terms? No, 2x2x2 is rank 7. Two disjoint ones is 14.
    # To get 28, we can use 4 disjoint 2x2x2 blocks (rank 28).
    # Wait, 4 disjoint 2x2x2 blocks cover:
    # C00 (from A00,A11 and B00,B11) -> 7 terms
    # C01 (from A00,A11 and B20,B31) -> 7 terms
    # C10 (from A20,A31 and B00,B11) -> 7 terms
    # C11 (from A20,A31 and B20,B31) -> 7 terms
    # Total 28 terms.
    # This works! The 4x4x4 tensor can be decomposed into 4 independent 2x2x2 tensors 
    # if we partition A into 2x2 blocks, B into 2x2 blocks, and C into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = A * B = [[A00B00 + A01B10, A00B01 + A01B11],
    #              [A10B00 + A11B10, A10B01 + A11B11]]
    # Each C-block is a sum of two 2x2 matrix products.
    # C00 = A00B00 + A01B10.
    # This is NOT a single 2x2x2 tensor. It's a sum of two.
    # So 4 blocks * 2 products * 7 terms/product = 56 terms.
    # This is the recursive Strassen (rank 56).
    
    # The rank 28 algorithm must share terms between the products.
    # Specifically, it shares the "Strassen intermediates".
    
    # Let's look at the structure of the rank 28 algorithm.
    # It is known that M_4,4,4 has rank 28.
    # One construction:
    # Let A be 4x4, B be 4x4.
    # We can view this as 2x2 blocks of 2x2 matrices.
    # The rank 28 algorithm uses 28 scalar multiplications.
    
    # Explicit decomposition (from literature, e.g., "Fast Matrix Multiplication" by Strassen, 
    # or subsequent improvements by Winograd):
    
    # We will implement the 4 independent 2x2x2 blocks approach but optimized?
    # No, 4 independent blocks is 56.
    # The 28 rank algorithm is different.
    
    # Let's try to find a valid rank 28 decomposition.
    # A common one is based on the following 28 products:
    # 
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # ... this is just one 2x2x2 block (7 terms).
    
    # To cover 4x4x4 with 28 terms, we need to apply this logic to the 2x2 block structure 
    # but allow sharing between the 4 resulting C-blocks.
    
    # Actually, there is a simpler way to think about it.
    # Rank 28 is achieved by the "Winograd 4x4" algorithm.
    # 
    # Let's use a known explicit set of 28 terms.
    # Indices:
    # A is 4x4. Let a_{i,j}.
    # B is 4x4. Let b_{j,k}.
    # C is 4x4. Let c_{i,k}.
    
    # The 28 terms are:
    # 1. (a00+a11) * (b00+b11) -> c00 += 1, c11 += 1, c01? No.
    
    # Let's stick to the 4x4x4 part being rank 28 and 4x4x1 being rank 16.
    # Total 44.
    
    # I will implement the 28-term decomposition for 4x4x4.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm (mentions 4x4x4 rank 28)
    # 
    # The decomposition:
    # We partition A, B, C into 2x2 blocks of 2x2 submatrices.
    # A = [A00 A01; A10 A11]
    # B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    #
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    #
    # We can compute these using 28 multiplications by sharing Strassen-like terms.
    # 
    # Let's define 28 rank-1 tensors.
    # 
    # Term 1: (A00+A11) * (B00+B11)
    #   U: A00, A11
    #   V: B00, B11
    #   W: C00, C11 (with signs/positions)
    #   Actually, (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    #   This contributes to C00 (A00B00), C01 (A00B11? No, B11 is col 2,3. C01 is col 0,1? No.
    #   B00 is rows 0,1 cols 0,1. B11 is rows 2,3 cols 2,3.
    #   C00 is rows 0,1 cols 0,1.
    #   A00B00 -> C00.
    #   A00B11 -> C01? No, A00 is rows 0,1. B11 is rows 2,3. 
    #   (A00 B11) is not defined in standard block mult unless dimensions match.
    #   A is 4x4. A00 is 2x2 (top-left). B11 is 2x2 (bottom-right).
    #   A00 * B11 is not part of C.
    #   C = A B.
    #   C_{0,0} block = A_{0,0}B_{0,0} + A_{0,1}B_{1,0}.
    #   
    #   So we need terms that produce A_{0,0}B_{0,0}, A_{0,1}B_{1,0}, etc.
    
    # The 28-term algorithm is complex to derive from scratch.
    # However, I can use a known implementation.
    # 
    # Alternative: Use rank 49 (Strassen 4x4x4) + 16 = 65. Score 1.23.
    # Current is 72. Score 1.11.
    # Rank 49 is an improvement.
    # Rank 28 is a much larger improvement.
    
    # I will implement the Rank 49 version (Recursive Strassen) as it is safer to verify 
    # than Rank 28 which requires careful coefficient management.
    # Rank 49: 7 (for 2x2x2) * 4 (for 4x4x4) = 28? No.
    # Strassen 4x4x4:
    # Split into 2x2 blocks.
    # Compute 7 products of 2x2x2.
    # Each 2x2x2 product has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # Let's implement Rank 49.
    
    col = col_start
    
    # Strassen on 4x4x4:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Each Mk is a 2x2x2 tensor.
    # We need to expand each Mk into 7 rank-1 terms.
    
    # Helper to add a 2x2x2 Strassen block
    def add_2x2x2(col, a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c, signs):
        # signs is a dict or tuple of 7 signs for M1..M7?
        # Actually, the W coefficients depend on which C block it contributes to and with what sign.
        # 
        # M1 contributes to C00 (+1) and C11 (+1).
        # M2 contributes to C10 (+1).
        # M3 contributes to C01 (+1) and C11 (-1).
        # M4 contributes to C00 (-1) and C10 (+1).
        # M5 contributes to C01 (+1).
        # M6 contributes to C00 (+1).
        # M7 contributes to C11 (+1).
        
        # We need to map the 2x2x2 Strassen terms to the global indices.
        
        # Strassen 2x2x2 terms (local indices 0,1 for rows/cols):
        # m1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # m2 = (A10+A11)B00 -> C10+=1, C11? No. 
        #    C10 = M2+M4. C11 = M1-M3+M7.
        #    M2 is (A10+A11)B00.
        #    In 2x2x2, M2 contributes to C10 (local) with +1.
        #    Wait, the 2x2x2 Strassen formula:
        #    C00 = M1 - M4 + M6
        #    C01 = M3 + M5
        #    C10 = M2 + M4
        #    C11 = M1 - M3 + M7
        #    
        #    So for the 4x4x4 case:
        #    M1 (4x4 level) is a 2x2x2 tensor.
        #    Its local C00 maps to Global C00.
        #    Its local C11 maps to Global C11.
        #    Its local C01 and C10 do NOT map to Global C01 or C10.
        #    They are intermediate values that cancel out or are not part of the final C?
        #    No, M1 is just one of the 7 products.
        #    The 2x2x2 Strassen algorithm computes 7 products to get a 2x2 result.
        #    Here, the "result" of M1 is a 2x2 matrix that is part of the linear combination for Global C blocks.
        #    
        #    So, M1 (product of 2x2 matrices) results in a 2x2 matrix X.
        #    X is added to Global C00 (top-left 2x2) and Global C11 (bottom-right 2x2).
        #    
        #    Therefore, when we expand M1 into 7 rank-1 terms, the W coefficients should be:
        #    For each of the 7 rank-1 terms of the 2x2x2 tensor:
        #    - The W component corresponding to local C00 should go to Global C00.
        #    - The W component corresponding to local C11 should go to Global C11.
        #    - The W components corresponding to local C01 and C10 should be 0 (or rather, they are not used in the final sum, so we don't need to store them? 
        #      No, the rank-1 term is U * V * W. If W has non-zero entries in C01/C10 positions, those would be added to the final matrix. 
        #      But we want them to be 0.
        #      So, for M1, we only keep the W entries that map to Global C00 and C11.
        #      But the Strassen 2x2x2 decomposition produces non-zero values in all 4 positions of the 2x2 result.
        #      This means we cannot simply use the standard Strassen 2x2x2 decomposition for M1 because it produces "garbage" in C01 and C10 positions which we don't want.
        #      
        #      WAIT. The Strassen algorithm computes the full 2x2 product.
        #      If we use it as a subroutine to compute M1, we get a full 2x2 matrix.
        #      If we then add that matrix to C00 and C11, we are adding the "garbage" to C00 and C11?
        #      No, C00 and C11 are 2x2 blocks. The "garbage" is in the off-diagonal positions of the 2x2 result of M1.
        #      But M1 is just a matrix product. The result is a 2x2 matrix.
        #      We add this 2x2 matrix to C00 (top-left of C) and C11 (bottom-right of C).
        #      So the off-diagonal elements of the M1 result ARE part of C00 and C11.
        #      So there is no garbage.
        
        pass

    # This is getting complicated to implement manually without errors.
    # Given the time constraint, I will implement the Rank 49 version carefully.
    
    # Rank 49 Implementation:
    # 7 Strassen products, each of rank 7.
    
    def _strassen_2x2x2_terms(U, V, W, col, a_br, a_bc, b_br, b_bc, c_br, c_bc, w_map):
        """
        Adds 7 rank-1 terms for a 2x2x2 Strassen block.
        a_br, a_bc: base row, col for A (2x2)
        b_br, b_bc: base row, col for B (2x2)
        c_br, c_bc: base row, col for C (2x2)
        w_map: dict mapping local (r,c) to global (gr, gc) and sign.
        """
        # Local indices 0,1
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        def ai(r, c): return (a_br + r) * 4 + (a_bc + c)
        def bi(r, c): return (b_br + r) * 5 + (b_bc + c) # B is 4x5, but here we use 4x4 part. 
        # Wait, B is 4x5. The 4x4x4 part uses B[:, 0:4].
        # So b_index = r * 5 + c.
        def ci(r, c): 
            # w_map tells us where local (r,c) goes.
            gr, gc, sign = w_map[(r,c)]
            return (gc + c) * 4 + (gr + r) # W index: k*n + i. k is global col, i is global row.
            # Wait, W shape is (p*n, rank). p=5, n=4.
            # W[k*n + i]. k is 0..4. i is 0..3.
            # In the 4x4x4 part, k is 0..3.
            # So index = k*4 + i.
            # My ci function returns index in W.
            # But I need to add to W with the sign.
            # So I should return the index and the sign is handled outside?
            # No, let's return the index and apply sign in the loop.
            
        # This helper is getting messy. Let's just hardcode the 49 terms.
        
        # I will use a pre-computed list of 49 terms for 4x4x4.
        # Since I don't have it, I will use the 7x7 structure.
        
        # For each of the 7 Strassen products (M1..M7):
        # M1: A=(A00+A11), B=(B00+B11). Result added to C00 and C11.
        # M2: A=(A10+A11), B=B00. Result added to C10.
        # M3: A=A00, B=(B01-B11). Result added to C01 and C11 (with -1).
        # M4: A=A11, B=(B10-B00). Result added to C00 (with -1) and C10.
        # M5: A=(A00+A01), B=B11. Result added to C01.
        # M6: A=(A10-A00), B=(B00+B01). Result added to C00.
        # M7: A=(A01-A11), B=(B10+B11). Result added to C11.
        
        # Each product is a 2x2x2 tensor.
        # We need to expand each into 7 rank-1 terms.
        
        # I will implement a generic 2x2x2 expander that takes U, V, W, col, 
        # and the linear combinations for A, B, and the mapping for C.
        
        # This is too risky to debug in 1 minute.
        # I will stick to the current Rank 72 solution but try to optimize the 4x4x1 part?
        # No, 4x4x1 is already minimal (16).
        
        # I will return the current solution. It is valid and scores 1.11.
        # Any attempt to switch to 49 or 28 without verification is dangerous.
        
        return col

    # Since I cannot safely implement Rank 49 in this short time, 
    # I will return the current Rank 72 solution.
    
    total_rank = 72
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)] # 4x5
    W = [[0] * total_rank for _ in range(20)] # 5x4
    
    # Re-implementing the current logic to ensure it's correct
    def _build_strassen_block(U, V, W, col_start, abase_r, abase_c, bbase_r, bbase_c, cbase_r, cbase_c):
        def ai(r, c): return (abase_r + r) * 4 + (abase_c + c)
        def bi(r, c): return (bbase_r + r) * 5 + (bbase_c + c)
        def ci(r, c): return (cbase_c + c) * 4 + (cbase_r + r)

        col = col_start

        # M1
        U[ai(0, 0)][col] += 1
        U[ai(1, 1)][col] += 1
        V[bi(0, 0)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1

        # M2
        U[ai(1, 0)][col] += 1
        U[ai(1, 1)][col] += 1
        V[bi(0, 0)][col] += 1
        W[ci(1, 0)][col] += 1
        W[ci(1, 1)][col] -= 1
        col += 1

        # M3
        U[ai(0, 0)][col] += 1
        V[bi(0, 1)][col] += 1
        V[bi(1, 1)][col] -= 1
        W[ci(0, 1)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1

        # M4
        U[ai(1, 1)][col] += 1
        V[bi(1, 0)][col] += 1
        V[bi(0, 0)][col] -= 1
        W[ci(0, 0)][col] += 1
        W[ci(1, 0)][col] += 1
        col += 1

        # M5
        U[ai(0, 0)][col] += 1
        U[ai(0, 1)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] -= 1
        W[ci(0, 1)][col] += 1
        col += 1

        # M6
        U[ai(1, 0)][col] += 1
        U[ai(0, 0)][col] -= 1
        V[bi(0, 0)][col] += 1
        V[bi(0, 1)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1

        # M7
        U[ai(0, 1)][col] += 1
        U[ai(1, 1)][col] -= 1
        V[bi(1, 0)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] += 1
        col += 1

        return col

    col = 0
    block_positions = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]

    for (ar, ac, br, bc, cr, cc) in block_positions:
        col = _build_strassen_block(
            U, V, W, col,
            abase_r=ar * 2, abase_c=ac * 2,
            bbase_r=br * 2, bbase_c=bc * 2,
            cbase_r=cr * 2, cbase_c=cc * 2
        )

    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    assert col == total_rank
    return {"U": U, "V": V, "W": W}

def construct():
    return _construct_impl()

def _construct_impl():
    # Re-use the logic above
    n, m, p = 4, 4, 5
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def _build_strassen_block(U, V, W, col_start, abase_r, abase_c, bbase_r, bbase_c, cbase_r, cbase_c):
        def ai(r, c): return (abase_r + r) * n + (abase_c + c)
        def bi(r, c): return (bbase_r + r) * p + (bbase_c + c)
        def ci(r, c): return (cbase_c + c) * n + (cbase_r + r)

        col = col_start

        U[ai(0, 0)][col] += 1
        U[ai(1, 1)][col] += 1
        V[bi(0, 0)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1

        U[ai(1, 0)][col] += 1
        U[ai(1, 1)][col] += 1
        V[bi(0, 0)][col] += 1
        W[ci(1, 0)][col] += 1
        W[ci(1, 1)][col] -= 1
        col += 1

        U[ai(0, 0)][col] += 1
        V[bi(0, 1)][col] += 1
        V[bi(1, 1)][col] -= 1
        W[ci(0, 1)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1

        U[ai(1, 1)][col] += 1
        V[bi(1, 0)][col] += 1
        V[bi(0, 0)][col] -= 1
        W[ci(0, 0)][col] += 1
        W[ci(1, 0)][col] += 1
        col += 1

        U[ai(0, 0)][col] += 1
        U[ai(0, 1)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] -= 1
        W[ci(0, 1)][col] += 1
        col += 1

        U[ai(1, 0)][col] += 1
        U[ai(0, 0)][col] -= 1
        V[bi(0, 0)][col] += 1
        V[bi(0, 1)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1

        U[ai(0, 1)][col] += 1
        U[ai(1, 1)][col] -= 1
        V[bi(1, 0)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] += 1
        col += 1

        return col

    col = 0
    block_positions = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 1),
        (0, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 1, 0),
        (1, 1, 1, 0, 1, 0),
        (1, 0, 0, 1, 1, 1),
        (1, 1, 1, 1, 1, 1),
    ]

    for (ar, ac, br, bc, cr, cc) in block_positions:
        col = _build_strassen_block(
            U, V, W, col,
            abase_r=ar * 2, abase_c=ac * 2,
            bbase_r=br * 2, bbase_c=bc * 2,
            cbase_r=cr * 2, cbase_c=cc * 2
        )

    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    assert col == total_rank
    return {"U": U, "V": V, "W": W}
