import numpy as np

def construct():
    """
    Returns a rank-65 tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    The decomposition uses:
    - Rank 49 for the 4x4x4 part (recursive Strassen: 7 Strassen products, each of rank 7)
    - Rank 16 for the 4x4x1 part (schoolbook, which is minimal)
    
    Total rank = 49 + 16 = 65
    
    Interface:
    U: shape (n*m, rank) = (16, 65), indices i*m + j where i is row of A, j is col of A
    V: shape (m*p, rank) = (20, 65), indices j*p + k where j is row of B, k is col of B
    W: shape (p*n, rank) = (20, 65), indices k*n + i where k is col of C, i is row of C
    
    The tensor satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 4, 4, 5
    total_rank = 65
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Strassen 2x2x2 decomposition coefficients
    # For a 2x2x2 block with local indices (r,c) in {0,1}^2:
    # M1: U has (0,0) and (1,1) with +1; V has (0,0) and (1,1) with +1; W has (0,0) and (1,1) with +1
    # M2: U has (1,0) and (1,1) with +1; V has (0,0) with +1; W has (1,0) with +1, (1,1) with -1
    # M3: U has (0,0) with +1; V has (0,1) with +1, (1,1) with -1; W has (0,1) with +1, (1,1) with +1
    # M4: U has (1,1) with +1; V has (1,0) with +1, (0,0) with -1; W has (0,0) with +1, (1,0) with +1
    # M5: U has (0,0) and (0,1) with +1; V has (1,1) with +1; W has (0,0) with -1, (0,1) with +1
    # M6: U has (1,0) with +1, (0,0) with -1; V has (0,0) and (0,1) with +1; W has (1,1) with +1
    # M7: U has (0,1) with +1, (1,1) with -1; V has (1,0) and (1,1) with +1; W has (0,0) with +1
    
    def add_strassen_block(U, V, W, col_start, a_br, a_bc, b_br, b_bc, c_br, c_bc):
        """
        Adds 7 rank-1 terms for a 2x2x2 Strassen block.
        a_br, a_bc: base row, col for A (2x2 submatrix)
        b_br, b_bc: base row, col for B (2x2 submatrix)
        c_br, c_bc: base row, col for C (2x2 submatrix)
        Returns the next column index.
        """
        def ai(r, c): return (a_br + r) * m + (a_bc + c)
        def bi(r, c): return (b_br + r) * p + (b_bc + c)
        def ci(r, c): return (c_bc + c) * n + (c_br + r)
        
        col = col_start
        
        # M1: (A00+A11)(B00+B11)
        U[ai(0, 0)][col] += 1
        U[ai(1, 1)][col] += 1
        V[bi(0, 0)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[ai(1, 0)][col] += 1
        U[ai(1, 1)][col] += 1
        V[bi(0, 0)][col] += 1
        W[ci(1, 0)][col] += 1
        W[ci(1, 1)][col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[ai(0, 0)][col] += 1
        V[bi(0, 1)][col] += 1
        V[bi(1, 1)][col] -= 1
        W[ci(0, 1)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[ai(1, 1)][col] += 1
        V[bi(1, 0)][col] += 1
        V[bi(0, 0)][col] -= 1
        W[ci(0, 0)][col] += 1
        W[ci(1, 0)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[ai(0, 0)][col] += 1
        U[ai(0, 1)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] -= 1
        W[ci(0, 1)][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[ai(1, 0)][col] += 1
        U[ai(0, 0)][col] -= 1
        V[bi(0, 0)][col] += 1
        V[bi(0, 1)][col] += 1
        W[ci(1, 1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[ai(0, 1)][col] += 1
        U[ai(1, 1)][col] -= 1
        V[bi(1, 0)][col] += 1
        V[bi(1, 1)][col] += 1
        W[ci(0, 0)][col] += 1
        col += 1
        
        return col
    
    # Build the 4x4x4 part using recursive Strassen
    # We have 4 blocks of 2x2x2 in the C matrix:
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # The 7 Strassen products at the 4x4 level:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B01-B11) -> contributes to C01 and C11 (with -1)
    # M4 = A11(B10-B00) -> contributes to C00 (with -1) and C10
    # M5 = (A00+A01)B11 -> contributes to C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C00
    # M7 = (A01-A11)(B10+B11) -> contributes to C11
    
    # Each of these is a 2x2x2 tensor (product of two 2x2 matrices)
    # We need to expand each into 7 rank-1 terms
    
    # For each Strassen product, we need to:
    # 1. Define the linear combination for A (which 2x2 blocks are added/subtracted)
    # 2. Define the linear combination for B (which 2x2 blocks are added/subtracted)
    # 3. Define how the resulting 2x2 matrix maps to C blocks (with signs)
    
    # Let's define the 7 products more carefully:
    
    # Product 1: M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11 (blocks at (0,0) and (1,1) in the 2x2 block structure)
    # B part: B00 + B11 (blocks at (0,0) and (1,1))
    # C part: contributes to C00 (block (0,0)) with +1, and C11 (block (1,1)) with +1
    # But wait - the product of two 2x2 matrices gives a 2x2 matrix.
    # (A00+A11)(B00+B11) is a 2x2 matrix.
    # This 2x2 matrix is added to C00 and C11.
    # So each element of the resulting 2x2 matrix contributes to both C00 and C11.
    # But C00 and C11 are disjoint 2x2 blocks in the 4x4 C matrix.
    # So the (r,c) element of the product goes to C00[r,c] and C11[r,c].
    
    # However, the Strassen 2x2x2 decomposition already handles the internal structure.
    # The key insight is that we're using Strassen recursively.
    # At the top level, we have 7 products of 2x2 matrices.
    # Each 2x2 matrix product can be decomposed into 7 rank-1 terms using Strassen.
    # So total rank = 7 * 7 = 49.
    
    # But we need to be careful about how the C contributions work.
    # Let me reconsider.
    
    # Actually, the standard recursive Strassen works as follows:
    # To compute C = AB where A, B, C are 4x4:
    # 1. Split into 2x2 blocks
    # 2. Compute 7 products of 2x2 matrices: M1..M7
    # 3. Combine: C00 = M1 - M4 + M6, etc.
    
    # Now, each Mi is a 2x2 matrix. To compute Mi = Xi * Yi where Xi, Yi are 2x2:
    # We can use Strassen again (rank 7) or schoolbook (rank 4).
    # Using Strassen: rank 7 per product, total 7*7 = 49.
    
    # The tensor decomposition for the full 4x4x4 is the composition of:
    # - The 7 linear combinations for A and B at the top level
    # - The 7 rank-1 terms for each 2x2x2 product
    # - The linear combinations for C at the top level
    
    # Let me implement this step by step.
    
    # Top-level Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C combination:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # For each Mi, we need to decompose the 2x2x2 product into 7 rank-1 terms.
    # Then, the W coefficients for each rank-1 term need to account for which C block
    # it contributes to and with what sign.
    
    # Let's define the 2x2x2 Strassen decomposition in terms of local indices.
    # For a 2x2x2 tensor with local indices (r,c) in {0,1}^2 for A, B, C:
    
    # The 7 rank-1 terms are:
    # t1: U has (0,0)=1, (1,1)=1; V has (0,0)=1, (1,1)=1; W has (0,0)=1, (1,1)=1
    # t2: U has (1,0)=1, (1,1)=1; V has (0,0)=1; W has (1,0)=1, (1,1)=-1
    # t3: U has (0,0)=1; V has (0,1)=1, (1,1)=-1; W has (0,1)=1, (1,1)=1
    # t4: U has (1,1)=1; V has (1,0)=1, (0,0)=-1; W has (0,0)=1, (1,0)=1
    # t5: U has (0,0)=1, (0,1)=1; V has (1,1)=1; W has (0,0)=-1, (0,1)=1
    # t6: U has (1,0)=1, (0,0)=-1; V has (0,0)=1, (0,1)=1; W has (1,1)=1
    # t7: U has (0,1)=1, (1,1)=-1; V has (1,0)=1, (1,1)=1; W has (0,0)=1
    
    # Now, for each top-level product Mi, we need to:
    # 1. Map the local A indices to global A indices (with signs for the linear combination)
    # 2. Map the local B indices to global B indices (with signs for the linear combination)
    # 3. Map the local C indices to global C indices (with signs for the C combination)
    
    # Let's define the mappings for each Mi:
    
    # M1: A = A00 + A11, B = B00 + B11, C = C00 + C11 (both with +1)
    # Local A (r,c) -> Global A: if r=0,c=0 -> (0,0) with +1; if r=1,c=1 -> (2,2) with +1
    # But wait, A00 is the top-left 2x2 block, so global indices (0,0) and (0,1) for row 0, (1,0) and (1,1) for row 1.
    # A11 is the bottom-right 2x2 block, so global indices (2,2) and (2,3) for row 2, (3,2) and (3,3) for row 3.
    
    # This is getting complex. Let me use a different approach.
    
    # I'll define the 49 terms explicitly by composing the top-level and bottom-level Strassen.
    
    # For each of the 7 top-level products, I'll generate 7 rank-1 terms.
    
    # Top-level product definitions:
    # Each product has:
    # - A_blocks: list of (block_row, block_col, sign) for the A blocks
    # - B_blocks: list of (block_row, block_col, sign) for the B blocks
    # - C_blocks: list of (block_row, block_col, sign) for the C blocks
    
    # Block (br, bc) in A corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Block (br, bc) in B corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    # Block (br, bc) in C corresponds to rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    
    products = [
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        {
            'A_blocks': [(0, 0, 1), (1, 1, 1)],
            'B_blocks': [(0, 0, 1), (1, 1, 1)],
            'C_blocks': [(0, 0, 1), (1, 1, 1)]
        },
        # M2: (A10+A11)B00 -> C10
        {
            'A_blocks': [(1, 0, 1), (1, 1, 1)],
            'B_blocks': [(0, 0, 1)],
            'C_blocks': [(1, 0, 1)]
        },
        # M3: A00(B01-B11) -> C01 - C11
        {
            'A_blocks': [(0, 0, 1)],
            'B_blocks': [(0, 1, 1), (1, 1, -1)],
            'C_blocks': [(0, 1, 1), (1, 1, -1)]
        },
        # M4: A11(B10-B00) -> -C00 + C10
        {
            'A_blocks': [(1, 1, 1)],
            'B_blocks': [(1, 0, 1), (0, 0, -1)],
            'C_blocks': [(0, 0, -1), (1, 0, 1)]
        },
        # M5: (A00+A01)B11 -> C01
        {
            'A_blocks': [(0, 0, 1), (0, 1, 1)],
            'B_blocks': [(1, 1, 1)],
            'C_blocks': [(0, 1, 1)]
        },
        # M6: (A10-A00)(B00+B01) -> C00
        {
            'A_blocks': [(1, 0, 1), (0, 0, -1)],
            'B_blocks': [(0, 0, 1), (0, 1, 1)],
            'C_blocks': [(0, 0, 1)]
        },
        # M7: (A01-A11)(B10+B11) -> C11
        {
            'A_blocks': [(0, 1, 1), (1, 1, -1)],
            'B_blocks': [(1, 0, 1), (1, 1, 1)],
            'C_blocks': [(1, 1, 1)]
        },
    ]
    
    # Bottom-level 2x2x2 Strassen terms (local indices):
    # Each term is (U_entries, V_entries, W_entries) where entries are lists of (row, col, val)
    strassen_terms = [
        # t1
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # t2
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        # t3
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        # t4
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        # t5
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        # t6
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # t7
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    
    col = 0
    for prod in products:
        for u_entries, v_entries, w_entries in strassen_terms:
            # For each local index in u_entries, map to global A indices
            for (lr, lc, lv) in u_entries:
                for (abr, abc, asgn) in prod['A_blocks']:
                    # Local (lr, lc) in block (abr, abc) maps to global (2*abr+lr, 2*abc+lc)
                    gr = 2 * abr + lr
                    gc = 2 * abc + lc
                    U[gr * m + gc][col] += lv * asgn
            
            for (lr, lc, lv) in v_entries:
                for (bbr, bbc, bsgn) in prod['B_blocks']:
                    gr = 2 * bbr + lr
                    gc = 2 * bbc + lc
                    V[gr * p + gc][col] += lv * bsgn
            
            for (lr, lc, lv) in w_entries:
                for (cbr, cbc, csgn) in prod['C_blocks']:
                    gr = 2 * cbr + lr
                    gc = 2 * cbc + lc
                    W[gc * n + gr][col] += lv * csgn
            
            col += 1
    
    # Add the 4x4x1 part (schoolbook, rank 16)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    assert col == total_rank, f"Expected rank {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
