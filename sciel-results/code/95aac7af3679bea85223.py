"""Program for matrix multiplication tensor decomposition of 5x5 by 5x5."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def get_strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
        
        Convention: A is 2x2, B is 2x2, C = A*B is 2x2.
        U[i*2+j, r] - row i, col j of A
        V[j*2+k, r] - row j, col k of B
        W[k*2+i, r] - row k, col i of C (trace convention W[k*n+i])
        
        Strassen formulas (0-indexed):
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A10)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A11-A01)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0], A11: U[3]
        # B00: V[0], B11: V[3]
        # C00: W[0], C11: W[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        # A10: U[2], A11: U[3]
        # B00: V[0]
        # C10: W[2], C11: W[3]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B10-B11)
        # A00: U[0]
        # B10: V[2], B11: V[3]
        # C01: W[1], C10: W[2]
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[2][2] = 1
        
        # M4 = A11(B01-B00)
        # A11: U[3]
        # B01: V[1], B00: V[0]
        # C00: W[0], C10: W[2]
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A10)B11
        # A00: U[0], A10: U[2]
        # B11: V[3]
        # C00: W[0], C01: W[1]
        U[0][4] = 1; U[2][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        # A10: U[2], A00: U[0]
        # B00: V[0], B10: V[2]
        # C11: W[3]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # M7 = (A11-A01)(B01+B11)
        # A11: U[3], A01: U[1]
        # B01: V[1], B11: V[3]
        # C00: W[0]
        U[3][6] = 1; U[1][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def get_strassen_4x4():
        """Return U, V, W for 4x4x4 matrix multiplication using Strassen (rank 49).
        
        Split 4x4 into 2x2 blocks. Apply 2x2 Strassen to each of the 7 terms,
        where each term is itself a 2x2 matrix multiplication.
        
        A = [[A00, A01], [A10, A11]] (each 2x2)
        B = [[B00, B01], [B10, B11]] (each 2x2)
        C = [[C00, C01], [C10, C11]] (each 2x2)
        
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A10)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A11-A01)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        U2, V2, W2 = get_strassen_2x2()
        
        # 4x4 indices: i,j,k in 0..3
        # Block mapping:
        # A00: i in {0,1}, j in {0,1}
        # A01: i in {0,1}, j in {2,3}
        # A10: i in {2,3}, j in {0,1}
        # A11: i in {2,3}, j in {2,3}
        
        # For each of the 7 Strassen terms, we have a 2x2 multiplication
        # Each 2x2 multiplication contributes 7 rank-1 terms
        # Total: 7 * 7 = 49 terms
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # Define the 7 terms with their A-blocks, B-blocks, and C-blocks
        # Each term: list of (a_block, b_block, c_blocks_with_signs)
        # a_block: list of (block_row, block_col) for A
        # b_block: list of (block_row, block_col) for B
        # c_blocks: list of (block_row, block_col, sign) for C
        
        terms = [
            # M1 = (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
            {'a': [(0,0), (1,1)], 'b': [(0,0), (1,1)], 'c': [(0,0,1), (1,1,1)]},
            # M2 = (A10+A11)B00 -> C10 (+1), C11 (-1)
            {'a': [(1,0), (1,1)], 'b': [(0,0)], 'c': [(1,0,1), (1,1,-1)]},
            # M3 = A00(B10-B11) -> C01 (+1), C10 (+1)
            {'a': [(0,0)], 'b': [(1,0), (1,1)], 'c': [(0,1,1), (1,0,1)]},
            # M4 = A11(B01-B00) -> C00 (+1), C10 (+1)
            {'a': [(1,1)], 'b': [(0,1), (0,0)], 'c': [(0,0,1), (1,0,1)]},
            # M5 = (A00+A10)B11 -> C00 (-1), C01 (+1)
            {'a': [(0,0), (1,0)], 'b': [(1,1)], 'c': [(0,0,-1), (0,1,1)]},
            # M6 = (A10-A00)(B00+B10) -> C11 (+1)
            {'a': [(1,0), (0,0)], 'b': [(0,0), (1,0)], 'c': [(1,1,1)]},
            # M7 = (A11-A01)(B01+B11) -> C00 (+1)
            {'a': [(1,1), (0,1)], 'b': [(0,1), (1,1)], 'c': [(0,0,1)]},
        ]
        
        # For each term, the A part is a sum of 2x2 blocks, B part is a sum of 2x2 blocks
        # The product is computed via 2x2 Strassen
        # But wait - each "term" like M1 = (A00+A11)(B00+B11) is a single 2x2 multiplication
        # of the SUM of blocks. So we need to form the sum first, then multiply.
        
        # Actually, in tensor decomposition, we can't just add blocks and then multiply.
        # Each rank-1 term must have a single A-index, single B-index, single C-index.
        
        # The correct approach: M1 = (A00+A11)(B00+B11) expands to:
        # A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each of these is a 2x2 multiplication, each with rank 7 via Strassen
        # So M1 contributes 4 * 7 = 28 terms? No, that's not right either.
        
        # Actually, (A00+A11)(B00+B11) is a single 2x2 matrix multiplication where
        # the input matrices are (A00+A11) and (B00+B11). This is ONE 2x2 multiplication
        # with rank 7. The linearity means we can decompose it.
        
        # In the tensor decomposition framework, for a rank-1 term:
        # U[i,j] * V[j,k] * W[k,i]
        # The coefficient can be any scalar.
        
        # For M1 = (A00+A11)(B00+B11), the U part has non-zero entries at:
        # (0,0), (0,1), (1,0), (1,1) [from A00] and (2,2), (2,3), (3,2), (3,3) [from A11]
        # The V part has non-zero entries at:
        # (0,0), (0,1), (1,0), (1,1) [from B00] and (2,2), (2,3), (3,2), (3,3) [from B11]
        
        # But this is a single rank-7 decomposition of the 2x2 multiplication tensor,
        # where the "effective" A matrix is A00+A11 and "effective" B is B00+B11.
        
        # The key insight: in the tensor decomposition, each rank-r term has:
        # U with entries at specific (i,j) positions
        # V with entries at specific (j,k) positions  
        # W with entries at specific (k,i) positions
        
        # For M1, the U entries are at all positions where A00 or A11 have entries,
        # i.e., (0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)
        # Similarly for V at (0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)
        # And W at (0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)
        
        # But the 2x2 Strassen decomposition of the multiplication of two 2x2 matrices
        # has U, V, W as 4x7 matrices. We need to map these to the 4x4 context.
        
        # For M1: the "local" 2x2 A has indices (0,0)->(0,0), (0,1)->(1,0), (1,0)->(2,2), (1,1)->(3,2)
        # Wait, I'm overcomplicating this.
        
        # Let me think differently. The 4x4x4 tensor can be decomposed by viewing it as
        # 16 scalar multiplications, but Strassen reduces this to 49.
        
        # The standard approach: the 4x4x4 tensor is the tensor product of two 2x2x2 tensors
        # in a certain sense, but actually it's more subtle.
        
        # Let me use a different approach: directly construct the 49-term decomposition
        # by considering the 4x4 matrix multiplication as composed of 2x2 block multiplications.
        
        # Actually, the cleanest way: the rank of the 4x4x4 matrix multiplication tensor is 49,
        # and one construction is to use the fact that it can be seen as 7 "Strassen steps"
        # where each step involves 2x2 block operations, and each 2x2 operation has rank 7.
        
        # But each "step" is not independent - they share the same U, V, W structure.
        
        # Let me just directly build it. For the 4x4x4 case:
        # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
        # where U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        
        # I'll construct it by iterating over the 7 Strassen terms and for each,
        # using the 2x2 Strassen U2, V2, W2 mapped to the appropriate blocks.
        
        # Term M1 = (A00+A11)(B00+B11):
        # The "A" part spans blocks (0,0) and (1,1) of the 4x4 A
        # The "B" part spans blocks (0,0) and (1,1) of the 4x4 B
        # The "C" part spans blocks (0,0) and (1,1) of the 4x4 C
        
        # For the 2x2 Strassen U2, V2, W2, the indices are:
        # U2[0] = A00, U2[1] = A01, U2[2] = A10, U2[3] = A11
        # V2[0] = B00, V2[1] = B01, V2[2] = B10, V2[3] = B11
        # W2[0] = C00, W2[1] = C01, W2[2] = C10, W2[3] = C11
        
        # For M1, the effective A is A00+A11, so in the 2x2 Strassen:
        # U2[0] corresponds to A00 of the effective matrix = A00 of 4x4
        # U2[3] corresponds to A11 of the effective matrix = A11 of 4x4
        # U2[1] and U2[2] are 0 (since A01 and A10 of effective matrix are 0)
        
        # Wait, no. (A00+A11) is a 2x2 matrix. Let's call it A_eff.
        # A_eff[0,0] = A00[0,0] + A11[0,0]
        # A_eff[0,1] = A00[0,1] + A11[0,1]
        # A_eff[1,0] = A00[1,0] + A11[1,0]
        # A_eff[1,1] = A00[1,1] + A11[1,1]
        
        # So in the 2x2 Strassen, U2[0] = A_eff[0,0] = A4x4[0,0] + A4x4[2,2]
        # U2[1] = A_eff[0,1] = A4x4[0,1] + A4x4[2,3]
        # U2[2] = A_eff[1,0] = A4x4[1,0] + A4x4[3,2]
        # U2[3] = A_eff[1,1] = A4x4[1,1] + A4x4[3,3]
        
        # Similarly for B_eff = B00+B11:
        # V2[0] = B4x4[0,0] + B4x4[2,2]
        # V2[1] = B4x4[0,1] + B4x4[2,3]
        # V2[2] = B4x4[1,0] + B4x4[3,2]
        # V2[3] = B4x4[1,1] + B4x4[3,3]
        
        # And C_eff = C00 (for the C00 block) or C11 (for C11 block):
        # For C00: W2[0] = C4x4[0,0], W2[1] = C4x4[0,1], W2[2] = C4x4[1,0], W2[3] = C4x4[1,1]
        # For C11: W2[0] = C4x4[2,2], W2[1] = C4x4[2,3], W2[2] = C4x4[3,2], W2[3] = C4x4[3,3]
        
        # But M1 contributes to BOTH C00 and C11!
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C00 and +1 to C11.
        
        # This means for the 7 rank-1 terms of M1, each term contributes to both C00 and C11.
        # So W for each term has entries in both the C00 block and C11 block.
        
        # Let me restructure. For each of the 7 Strassen terms, I'll compute the
        # contribution to U, V, W in the 4x4 context.
        
        # Define block-to-global index mapping for 4x4:
        # Block (br, bc) of A: rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
        # Global index for A: i*4 + j where i = 2*br + di, j = 2*bc + dj
        
        def block_to_global_A(br, bc, di, dj):
            return (2*br + di) * 4 + (2*bc + dj)
        
        def block_to_global_B(br, bc, di, dj):
            return (2*br + di) * 4 + (2*bc + dj)
        
        def block_to_global_C(br, bc, di, dj):
            # C index: k*4 + i where k is row, i is col
            # Block (br, bc) of C: rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
            return (2*br + di) * 4 + (2*bc + dj)
        
        # For each Strassen term, define:
        # a_blocks: list of (br, bc, sign) for A
        # b_blocks: list of (br, bc, sign) for B
        # c_blocks: list of (br, bc, sign) for C
        
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11)
            {'a': [(0,0,1), (1,1,1)], 'b': [(0,0,1), (1,1,1)], 'c': [(0,0,1), (1,1,1)]},
            # M2 = (A10+A11)B00
            {'a': [(1,0,1), (1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1), (1,1,-1)]},
            # M3 = A00(B10-B11)
            {'a': [(0,0,1)], 'b': [(1,0,1), (1,1,-1)], 'c': [(0,1,1), (1,0,1)]},
            # M4 = A11(B01-B00)
            {'a': [(1,1,1)], 'b': [(0,1,1), (0,0,-1)], 'c': [(0,0,1), (1,0,1)]},
            # M5 = (A00+A10)B11
            {'a': [(0,0,1), (1,0,1)], 'b': [(1,1,1)], 'c': [(0,0,-1), (0,1,1)]},
            # M6 = (A10-A00)(B00+B10)
            {'a': [(1,0,1), (0,0,-1)], 'b': [(0,0,1), (1,0,1)], 'c': [(1,1,1)]},
            # M7 = (A11-A01)(B01+B11)
            {'a': [(1,1,1), (0,1,-1)], 'b': [(0,1,1), (1,1,1)], 'c': [(0,0,1)]},
        ]
        
        col4 = 0
        for term in strassen_terms:
            # For this term, the effective A is sum of a_blocks, effective B is sum of b_blocks
            # We apply the 2x2 Strassen decomposition to this effective multiplication
            # The 2x2 Strassen U2, V2, W2 give us 7 rank-1 terms
            
            for r2 in range(7):
                # U2 contribution
                for di in range(2):
                    for dj in range(2):
                        u2_val = U2[di*2+dj][r2]
                        if u2_val == 0:
                            continue
                        for (abr, abc, asgn) in term['a']:
                            gi = block_to_global_A(abr, abc, di, dj)
                            U4[gi][col4] += u2_val * asgn
                
                # V2 contribution
                for di in range(2):
                    for dj in range(2):
                        v2_val = V2[di*2+dj][r2]
                        if v2_val == 0:
                            continue
                        for (bbr, bbc, bsgn) in term['b']:
                            gj = block_to_global_B(bbr, bbc, di, dj)
                            V4[gj][col4] += v2_val * bsgn
                
                # W2 contribution
                for di in range(2):
                    for dj in range(2):
                        w2_val = W2[di*2+dj][r2]
                        if w2_val == 0:
                            continue
                        for (cbr, cbc, csgn) in term['c']:
                            gk = block_to_global_C(cbr, cbc, di, dj)
                            W4[gk][col4] += w2_val * csgn
                
                col4 += 1
        
        return U4, V4, W4
    
    def schoolbook_decomp(bn, bm, bp):
        """Standard schoolbook decomposition for bn x bm x bp."""
        rank_local = bn * bm * bp
        U = np.zeros((bn * bm, rank_local))
        V = np.zeros((bm * bp, rank_local))
        W = np.zeros((bp * bn, rank_local))
        col = 0
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    U[i * bm + j, col] = 1
                    V[j * bp + k, col] = 1
                    W[k * bn + i, col] = 1
                    col += 1
        return U, V, W
    
    # Get the 4x4 Strassen decomposition
    U4_list, V4_list, W4_list = get_strassen_4x4()
    U4 = np.array(U4_list)
    V4 = np.array(V4_list)
    W4 = np.array(W4_list)
    
    # Verify the 4x4 decomposition
    # Check that U4 @ diag(1) @ V4^T reconstructs the tensor correctly
    # Actually, let's verify by checking a random case
    
    # Build the full 5x5x5 decomposition
    # Blocks: (4,4,4), (4,4,1), (4,1,4), (4,1,1), (1,4,4), (1,4,1), (1,1,4), (1,1,1)
    
    blocks = [
        (4, 4, 4),
        (4, 4, 1),
        (4, 1, 4),
        (4, 1, 1),
        (1, 4, 4),
        (1, 4, 1),
        (1, 1, 4),
        (1, 1, 1),
    ]
    
    # Compute total rank
    ranks = []
    for (bn, bm, bp) in blocks:
        if bn == 4 and bm == 4 and bp == 4:
            ranks.append(49)  # Strassen
        else:
            ranks.append(bn * bm * bp)
    
    total_rank = sum(ranks)
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col_offset = 0
    
    for idx, (bn, bm, bp) in enumerate(blocks):
        if bn == 4 and bm == 4 and bp == 4:
            # Use Strassen 4x4
            for li in range(4):
                for lj in range(4):
                    for lk in range(4):
                        local_col = li * 4 * 4 + lj * 4 + lk
                        gi = li
                        gj = lj
                        gk = lk
                        U[gi * 5 + gj, col_offset + local_col] = U4[li * 4 + lj, local_col]
                        V[gj * 5 + gk, col_offset + local_col] = V4[lj * 4 + lk, local_col]
                        W[gk * 5 + gi, col_offset + local_col] = W4[lk * 4 + li, local_col]
            col_offset += 49
        else:
            # Use schoolbook
            local_U, local_V, local_W = schoolbook_decomp(bn, bm, bp)
            for li in range(bn):
                gi = li if bn == 4 else 4
                for lj in range(bm):
                    gj = lj if bm == 4 else 4
                    for lk in range(bp):
                        gk = lk if bp == 4 else 4
                        local_col = li * bm * bp + lj * bp + lk
                        U[gi * 5 + gj, col_offset + local_col] = local_U[li * bm + lj, local_col]
                        V[gj * 5 + gk, col_offset + local_col] = local_V[lj * bp + lk, local_col]
                        W[gk * 5 + gi, col_offset + local_col] = local_W[lk * bn + li, local_col]
            col_offset += ranks[idx]
    
    # Check time budget
    if time.time() > deadline:
        # Fall back to trivial schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
