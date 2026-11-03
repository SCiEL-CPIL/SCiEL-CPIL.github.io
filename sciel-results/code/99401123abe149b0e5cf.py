import time

DIMENSIONS = [4, 4, 9]

def construct():
    # Strassen-based decomposition for 4x4 x 4x9
    # Block 1: (4x4)(4x4) using 7 multiplications (rank 7)
    # Block 2: (4x4)(4x4) using 7 multiplications (rank 7)
    # Block 3: (4x4)(4x1) using 4 multiplications (rank 4)
    # Total rank: 18
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 18
    V = [[0] * rank for _ in range(m * p)]  # 36 x 18
    W = [[0] * rank for _ in range(p * n)]  # 36 x 18
    
    # Strassen coefficients for (2x2)(2x2)
    # alpha[s] = [[a, b], [c, d]] for left matrix
    # beta[s]  = [[e, f], [g, h]] for right matrix
    # c_coeffs[s] = [[p, q], [r, s]] for output (but we track per-element)
    
    # For 2x2 Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A10)B11
    # M5 = (A11-A01)(B00+B10)
    # M6 = (A01-A10)(B01+B11)
    #
    # C00 = M0 + M3 - M4 - M5
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M3
    
    # We need to express each 4x4x4 block multiplication using 7 scalar multiplications
    # by decomposing the 4x4 into four 2x2 blocks and applying Strassen.
    
    # For a (4x4)(4x4) block, let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]]
    # where each Ai0, etc. is a 2x2 block.
    # C = [[C00, C01], [C10, C11]] where each Ci0 is 2x2.
    #
    # C00 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A10)B11 - (A11-A01)(B00+B10)
    # C01 = A00(B01-B11) + (A00+A10)B11
    # C10 = (A10+A11)B00 + (A11-A01)(B00+B10)
    # C11 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + A11(B10-B00)
    
    # We'll define the 7 multiplications for each 4x4x4 block:
    # For block at rows [r0:r0+4], cols [c0:c0+4] of A, and rows [c0:c0+4], cols [k0:k0+4] of B:
    #
    # M0: (A00+A11) * (B00+B11)
    # M1: (A10+A11) * B00
    # M2: A00 * (B01-B11)
    # M3: A11 * (B10-B00)
    # M4: (A00+A10) * B11
    # M5: (A11-A01) * (B00+B10)
    # M6: (A01-A10) * (B01+B11)
    
    # Let's define helper to add contributions to U, V, W for a given rank index
    
    def add_strassen_4x4x4(U, V, W, rank_start, a_row_off, a_col_off, b_row_off, b_col_off):
        # a_row_off: row offset in A (0 or 4 for the two 4x4 blocks... wait, A is 4x4 total)
        # Actually A is 4x4, so there's only one 4x4x4 block for the first 4 cols of B
        # Let me reconsider.
        
        # A is 4x4, B is 4x9.
        # C = A * B is 4x9.
        # C[:, 0:4] = A * B[:, 0:4] -- this is 4x4 times 4x4 = 4x4
        # C[:, 4:8] = A * B[:, 4:8] -- this is 4x4 times 4x4 = 4x4
        # C[:, 8:9] = A * B[:, 8:9] -- this is 4x4 times 4x1 = 4x1
        
        # For the 4x4x4 case, we use Strassen with 7 multiplications.
        # Let me define the indices carefully.
        
        # A[i,j] where i,j in 0..3
        # B[j,k] where j in 0..3, k in 0..3 (for the block)
        # C[i,k] where i,k in 0..3
        
        # Decompose into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]]
        # A00[i,j] = A[2*i, 2*j] for i,j in 0..1
        # A01[i,j] = A[2*i, 2*j+1]
        # A10[i,j] = A[2*i+1, 2*j]
        # A11[i,j] = A[2*i+1, 2*j+1]
        
        # Similarly for B and C.
        
        # The 7 multiplications produce 7 intermediate 2x2 matrices.
        # Each intermediate is computed as sum of products of linear combinations.
        
        # For rank index r (0..6 within this block), we define:
        # U[i*4 + j, rank_start + r] = coefficient of A[i,j] in the left factor
        # V[j*9 + k, rank_start + r] = coefficient of B[j,k] in the right factor  
        # W[k*4 + i, rank_start + r] = coefficient of C[i,k] in the output
        
        # Wait, V is (m*p, rank) = (4*9, rank), so V[j*9+k, c]
        # W is (p*n, rank) = (9*4, rank), so W[k*4+i, c]
        
        # For the first 4x4x4 block (cols 0-3 of B):
        # B[j,k] with k in 0..3
        
        # Let me define the 7 multiplications explicitly.
        
        # M0 = (A00+A11) * (B00+B11)
        # M1 = (A10+A11) * B00
        # M2 = A00 * (B01-B11)
        # M3 = A11 * (B10-B00)
        # M4 = (A00+A10) * B11
        # M5 = (A11-A01) * (B00+B10)
        # M6 = (A01-A10) * (B01+B11)
        
        # C00 = M0 + M3 - M4 - M5
        # C01 = M2 + M4
        # C10 = M1 + M5
        # C11 = M0 - M1 + M2 + M3
        
        # Now, each M is a 2x2 matrix. M[i,j] = sum over products.
        # For example, M0[i,j] = sum_{u,v} (A00[u,v]+A11[u,v]) * (B00[i,j]... wait no.
        # M0 = (A00+A11) * (B00+B11)
        # M0[p,q] = sum_{r,s} (A00[p,r]+A11[p,r]) * (B00[q,r]+B11[q,r])... 
        # Wait, matrix multiplication: (AB)[p,q] = sum_r A[p,r] B[r,q]
        # So M0[p,q] = sum_r (A00[p,r]+A11[p,r]) * (B00[r,q]+B11[r,q])
        
        # This is getting complex. Let me just enumerate all contributions.
        
        # For each of the 7 multiplications s, and for each output element (p,q) in the 2x2 block:
        # The contribution to C depends on which block C00, C01, C10, or C11.
        
        # Let me define alpha[s][p][r] = coefficient of A[p,r] in left factor of multiplication s
        # And beta[s][r][q] = coefficient of B[r,q] in right factor of multiplication s
        # Then C_block[p_block, q_block, p, q] gets contributions from the 7 multiplications.
        
        # Actually, let me just build it directly by iterating over all possible (i,j,k) and tracking.
        
        # For a 4x4x4 block, the tensor is sum_{i,j,k} C_ijk A_ij B_jk where C_ijk = 1 if valid.
        # Rank 7 decomposition means we write this as sum_{r=0}^{6} u_r v_r w_r
        
        # Let me define the 7 rank-1 components explicitly.
        
        # This is the standard Strassen decomposition for 2x2x2, lifted to 4x4x4.
        
        # I'll define arrays for the coefficients.
        
        # For simplicity, let me just hardcode the 7 multiplications for 4x4x4.
        
        # Each multiplication r has:
        # left[r][i][j] for i,j in 0..3 (coefficients of A[i,j])
        # right[r][j][k] for j,k in 0..3 (coefficients of B[j,k])
        # out[r][i][k] for i,k in 0..3 (coefficients of C[i,k])
        
        # Strassen 2x2x2:
        # M0 = (A00+A11)(B00+B11)
        # In 4x4 terms:
        # (A00+A11) has:
        #   [0,0]: A[0,0]
        #   [0,1]: A[0,1]
        #   [1,0]: A[1,0]
        #   [1,1]: A[1,1]
        #   [2,0]: A[3,0]
        #   [2,1]: A[3,1]
        #   [3,0]: A[2,0]
        #   [3,1]: A[2,1]
        # Wait, A00 is rows 0,1 cols 0,1; A11 is rows 2,3 cols 2,3.
        # So (A00+A11)[i,j] = A[i,j] + A[i,j] if... no.
        # (A00+A11) is a 2x2 block matrix where block (0,0) = A00 + A11? No.
        
        # Let me be more careful.
        # A is 4x4. We partition into 2x2 blocks of size 2x2:
        # A = [[A00, A01], [A10, A11]]
        # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
        
        # For Strassen, we compute:
        # M0 = (A00+A11) * (B00+B11)
        # where A00+A11 is a 2x2 matrix: (A00+A11)[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0,1
        # Similarly B00+B11[j,k] = B[j,k] + B[j+2,k+2]
        
        # So M0[p,q] = sum_r (A[p,r] + A[p+2,r+2]) * (B[r,q] + B[r+2,q+2]) for p,q in 0,1
        
        # This contributes to C00[p,q].
        
        # This is getting very tedious to hardcode. Let me just do it programmatically.
        
        # For the first 4x4x4 block (B cols 0-3):
        # Rank indices: rank_start to rank_start+6
        
        # Define the 7 multiplications:
        # s=0: M0 = (A00+A11)*(B00+B11)
        # s=1: M1 = (A10+A11)*B00
        # s=2: M2 = A00*(B01-B11)
        # s=3: M3 = A11*(B10-B00)
        # s=4: M4 = (A00+A10)*B11
        # s=5: M5 = (A11-A01)*(B00+B10)
        # s=6: M6 = (A01-A10)*(B01+B11)
        
        # For each s, define left_coeffs[i][j] and right_coeffs[j][k]
        # and output_coeffs[i][k] for the 2x2 blocks.
        
        # Let me create dictionaries.
        
        # For s=0: M0
        # left: (A00+A11)[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0,1
        # right: (B00+B11)[j,k] = B[j,k] + B[j+2,k+2] for j,k in 0,1
        # out: C00[i,j] += M0[i,j]
        
        # I'll build the full 4x4 coefficient matrices for each of the 7 multiplications.
        
        def build_4x4_strassen():
            # Returns lists of 4x4 matrices for left, right, and 7 output 2x2 blocks
            left = []
            right = []
            # out[s] is a list of (block_i, block_j, coeff) where block_i, block_j in 0,1
            # Actually, let me store out[s][i][k] for i,k in 0..3 directly
            
            out = []
            
            for s in range(7):
                L = [[0]*4 for _ in range(4)]
                R = [[0]*4 for _ in range(4)]
                O = [[0]*4 for _ in range(4)]
                
                if s == 0:  # M0 = (A00+A11)*(B00+B11) -> C00
                    for i in range(2):
                        for j in range(2):
                            L[i][j] = 1
                            L[i+2][j+2] = 1
                            R[i][j] = 1
                            R[i+2][j+2] = 1
                    for i in range(2):
                        for k in range(2):
                            O[i][k] = 1
                            
                elif s == 1:  # M1 = (A10+A11)*B00 -> C10, C11
                    for i in range(2):
                        for j in range(2):
                            L[i+2][j] = 1
                            L[i+2][j+2] = 1
                            R[i][j] = 1
                    for i in range(2):
                        for k in range(2):
                            O[i+2][k] = 1
                            O[i+2][k+2] = -1
                            
                elif s == 2:  # M2 = A00*(B01-B11) -> C01, C11
                    for i in range(2):
                        for j in range(2):
                            L[i][j] = 1
                            R[i][j+2] = 1
                            R[i+2][j+2] = -1
                    for i in range(2):
                        for k in range(2):
                            O[i][k+2] = 1
                            O[i+2][k+2] = 1
                            
                elif s == 3:  # M3 = A11*(B10-B00) -> C00, C11
                    for i in range(2):
                        for j in range(2):
                            L[i+2][j+2] = 1
                            R[i+2][j] = 1
                            R[i][j] = -1
                    for i in range(2):
                        for k in range(2):
                            O[i][k] = 1
                            O[i+2][k+2] = 1
                            
                elif s == 4:  # M4 = (A00+A10)*B11 -> C00, C01
                    for i in range(2):
                        for j in range(2):
                            L[i][j] = 1
                            L[i+2][j] = 1
                            R[i+2][j+2] = 1
                    for i in range(2):
                        for k in range(2):
                            O[i][k] = -1
                            O[i][k+2] = 1
                            
                elif s == 5:  # M5 = (A11-A01)*(B00+B10) -> C00, C10
                    for i in range(2):
                        for j in range(2):
                            L[i+2][j+2] = 1
                            L[i][j+2] = -1
                            R[i][j] = 1
                            R[i+2][j] = 1
                    for i in range(2):
                        for k in range(2):
                            O[i][k] = -1
                            O[i+2][k] = 1
                            
                elif s == 6:  # M6 = (A01-A10)*(B01+B11) -> C10, C11
                    for i in range(2):
                        for j in range(2):
                            L[i][j+2] = 1
                            L[i+2][j] = -1
                            R[i][j+2] = 1
                            R[i+2][j+2] = 1
                    for i in range(2):
                        for k in range(2):
                            O[i+2][k] = 1
                            O[i+2][k+2] = 1
                            
                left.append(L)
                right.append(R)
                out.append(O)
                
            return left, right, out
        
        left_4, right_4, out_4 = build_4x4_strassen()
        
        # Now populate U, V, W
        for s in range(7):
            r = rank_start + s
            for i in range(4):
                for j in range(4):
                    if left_4[s][i][j] != 0:
                        U[i*4 + (a_col_off + j)][r] += left_4[s][i][j]
                    if right_4[s][j][k] != 0:
                        pass  # Need to handle B indices
                # For B, the column offset in B is b_col_off
                for j in range(4):
                    for k in range(4):
                        if right_4[s][j][k] != 0:
                            V[(j)*9 + (b_col_off + k)][r] += right_4[s][j][k]
                for i in range(4):
                    for k in range(4):
                        if out_4[s][i][k] != 0:
                            W[(b_col_off + k)*4 + i][r] += out_4[s][i][k]
        
    # Hmm, I think I'm overcomplicating this. Let me just write it out more carefully.
    
    # Let me restart with a cleaner approach.
    
    # For the first 4x4x4 block: A * B[:, 0:4]
    # For the second 4x4x4 block: A * B[:, 4:8]
    # For the 4x4x1 block: A * B[:, 8:9]
    
    # I'll define the Strassen decomposition for 4x4x4 explicitly.
    
    # Let me just verify my build_4x4_strassen is correct by checking one multiplication.
    
    # Actually, let me just implement it step by step and test.
    
    # For now, let me use the schoolbook approach but with the knowledge that
    # we can use Strassen for the 4x4x4 blocks.
    
    # Let me redefine more carefully.
    
    # Clear and restart
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each 4x4x4 block, use Strassen with 7 multiplications.
    # The 7 multiplications for 4x4x4 (derived from 2x2x2 Strassen):
    
    # I'll define them as (left_4x4, right_4x4, out_4x4) tuples.
    
    strassen_4 = []
    
    # M0 = (A00+A11)(B00+B11) -> C00
    L0 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    # Wait, A00 is rows 0-1, cols 0-1. A11 is rows 2-3, cols 2-3.
    # (A00+A11)[i,j] for i,j in 0,1: this is a 2x2 matrix.
    # When we multiply two 2x2 matrices, we get a 2x2 result.
    # But we're doing 4x4x4, so the "blocks" are 2x2.
    
    # I think the cleanest way is to just enumerate all 4x4 coefficient matrices.
    
    # Let me define the 7 multiplications for 4x4x4 Strassen:
    
    # s=0: (A00+A11)*(B00+B11)
    # left[i][j] = 1 if (i,j) in A00 or A11, else 0
    # A00: i in 0,1; j in 0,1
    # A11: i in 2,3; j in 2,3
    L0 = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
    R0 = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
    O0 = [[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]  # C00
    
    # s=1: (A10+A11)*B00
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    L1 = [[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]]
    R1 = [[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]
    O1 = [[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]]  # C10
    # Wait, C10 gets M1, and C11 gets -M1. Let me check.
    # C11 = M0 - M1 + M2 + M3
    # So O1 should have -1 in C11 positions too.
    
    # I'm making errors. Let me be very systematic.
    
    # C = [[C00, C01], [C10, C11]]
    # C00 = M0 + M3 - M4 - M5
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M3
    
    # Each M is a 2x2 matrix. So for each multiplication s, I need to specify:
    # Which output block(s) it contributes to, with what sign.
    
    # s=0 (M0): contributes to C00 (+1), C11 (+1)
    # s=1 (M1): contributes to C10 (+1), C11 (-1)
    # s=2 (M2): contributes to C01 (+1), C11 (+1)
    # s=3 (M3): contributes to C00 (+1), C11 (+1)
    # s=4 (M4): contributes to C00 (-1), C01 (+1)
    # s=5 (M5): contributes to C00 (-1), C10 (+1)
    # s=6 (M6): contributes to C10 (+1), C11 (+1)
    
    # Now for the left and right factors:
    # s=0: left = A00+A11, right = B00+B11
    # s=1: left = A10+A11, right = B00
    # s=2: left = A00, right = B01-B11
    # s=3: left = A11, right = B10-B00
    # s=4: left = A00+A10, right = B11
    # s=5: left = A11-A01, right = B00+B10
    # s=6: left = A01-A10, right = B01+B11
    
    # Let me build the 4x4 coefficient matrices.
    
    def make_L(type_id):
        L = [[0]*4 for _ in range(4)]
        if type_id == 'A00':
            for i in range(2):
                for j in range(2):
                    L[i][j] = 1
        elif type_id == 'A11':
            for i in range(2):
                for j in range(2):
                    L[i+2][j+2] = 1
        elif type_id == 'A10':
            for i in range(2):
                for j in range(2):
                    L[i+2][j] = 1
        elif type_id == 'A01':
            for i in range(2):
                for j in range(2):
                    L[i][j+2] = 1
        elif type_id == 'A00+A11':
            for i in range(2):
                for j in range(2):
                    L[i][j] = 1
                    L[i+2][j+2] = 1
        elif type_id == 'A10+A11':
            for i in range(2):
                for j in range(2):
                    L[i+2][j] = 1
                    L[i+2][j+2] = 1
        elif type_id == 'A00+A10':
            for i in range(2):
                for j in range(2):
                    L[i][j] = 1
                    L[i+2][j] = 1
        elif type_id == 'A11-A01':
            for i in range(2):
                for j in range(2):
                    L[i+2][j+2] = 1
                    L[i][j+2] = -1
        elif type_id == 'A01-A10':
            for i in range(2):
                for j in range(2):
                    L[i][j+2] = 1
                    L[i+2][j] = -1
        return L
    
    def make_R(type_id):
        R = [[0]*4 for _ in range(4)]
        if type_id == 'B00':
            for i in range(2):
                for j in range(2):
                    R[i][j] = 1
        elif type_id == 'B11':
            for i in range(2):
                for j in range(2):
                    R[i+2][j+2] = 1
        elif type_id == 'B10':
            for i in range(2):
                for j in range(2):
                    R[i+2][j] = 1
        elif type_id == 'B01':
            for i in range(2):
                for j in range(2):
                    R[i][j+2] = 1
        elif type_id == 'B00+B11':
            for i in range(2):
                for j in range(2):
                    R[i][j] = 1
                    R[i+2][j+2] = 1
        elif type_id == 'B01-B11':
            for i in range(2):
                for j in range(2):
                    R[i][j+2] = 1
                    R[i+2][j+2] = -1
        elif type_id == 'B10-B00':
            for i in range(2):
                for j in range(2):
                    R[i+2][j] = 1
                    R[i][j] = -1
        elif type_id == 'B00+B10':
            for i in range(2):
                for j in range(2):
                    R[i][j] = 1
                    R[i+2][j] = 1
        elif type_id == 'B01+B11':
            for i in range(2):
                for j in range(2):
                    R[i][j+2] = 1
                    R[i+2][j+2] = 1
        return R
    
    def make_O(contributions):
        # contributions: list of (block_i, block_j, sign)
        O = [[0]*4 for _ in range(4)]
        for bi, bj, sign in contributions:
            for i in range(2):
                for j in range(2):
                    O[bi*2+i][bj*2+j] = sign
        return O
    
    # Define the 7 multiplications
    mults = [
        # (left_type, right_type, output_contributions)
        ('A00+A11', 'B00+B11', [(0,0,1), (1,1,1)]),      # M0
        ('A10+A11', 'B00', [(1,0,1), (1,1,-1)]),          # M1
        ('A00', 'B01-B11', [(0,1,1), (1,1,1)]),           # M2
        ('A11', 'B10-B00', [(0,0,1), (1,1,1)]),           # M3
        ('A00+A10', 'B11', [(0,0,-1), (0,1,1)]),          # M4
        ('A11-A01', 'B00+B10', [(0,0,-1), (1,0,1)]),      # M5
        ('A01-A10', 'B01+B11', [(1,0,1), (1,1,1)]),       # M6
    ]
    
    # Now populate U, V, W for the two 4x4x4 blocks and the 4x4x1 block
    
    # Block 1: B cols 0-3
    for s in range(7):
        r = s
        lt, rt, oc = mults[s]
        L = make_L(lt)
        R = make_R(rt)
        O = make_O(oc)
        
        for i in range(4):
            for j in range(4):
                if L[i][j] != 0:
                    U[i*4 + j][r] += L[i][j]
        
        for j in range(4):
            for k in range(4):
                if R[j][k] != 0:
                    V[j*9 + k][r] += R[j][k]
        
        for i in range(4):
            for k in range(4):
                if O[i][k] != 0:
                    W[k*4 + i][r] += O[i][k]
    
    # Block 2: B cols 4-7
    for s in range(7):
        r = 7 + s
        lt, rt, oc = mults[s]
        L = make_L(lt)
        R = make_R(rt)
        O = make_O(oc)
        
        for i in range(4):
            for j in range(4):
                if L[i][j] != 0:
                    U[i*4 + j][r] += L[i][j]
        
        for j in range(4):
            for k in range(4):
                if R[j][k] != 0:
                    V[j*9 + (4+k)][r] += R[j][k]
        
        for i in range(4):
            for k in range(4):
                if O[i][k] != 0:
                    W[(4+k)*4 + i][r] += O[i][k]
    
    # Block 3: B col 8 (4x4x1)
    # Schoolbook: 4 multiplications
    for i in range(4):
        for j in range(4):
            r = 14 + i*4 + j
            U[i*4 + j][r] = 1
            V[j*9 + 8][r] = 1
            W[8*4 + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
