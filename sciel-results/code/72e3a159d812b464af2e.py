"""Matrix multiplication 4x4x9 with Strassen-based block decomposition."""
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin

    n, m, p = DIMENSIONS  # 4, 4, 9

    # We'll build U, V, W matrices.
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 36 x rank
    # W: (p*n) x rank = 36 x rank

    # Strategy: Split into two 4x4x4 subproblems and one 4x4x1 subproblem.
    # For 4x4x4: use Strassen's rank-7 for 2x2x2, extended to 4x4x4.
    # Actually, let's use a known rank-49 algorithm for 4x4x4 (Strassen-based).
    # For 4x4x1: schoolbook rank-16.
    # Total rank: 49 + 49 + 16 = 114.

    # But wait, the known best for 4x4x4 is actually rank 49 (Strassen).
    # Let me construct this properly.

    # For 4x4x4 matrix multiplication, we can use the following approach:
    # Split each 4x4 matrix into 2x2 blocks.
    # Use Strassen's 7-multiplication algorithm for 2x2x2.
    # This gives rank 49 for 4x4x4 (since 7^2 = 49 scalar multiplications).

    # Actually, let me think more carefully. For 4x4x4:
    # We have A (4x4) * B (4x4) = C (4x4)
    # Split into 2x2 blocks: A = [[A11, A12], [A21, A22]], etc.
    # Using Strassen for each 2x2 block multiplication:
    # The number of scalar multiplications is 7 * 7 = 49? No, that's not right.
    # 
    # Actually, for 4x4x4, if we use a hierarchical Strassen:
    # Level 1: 7 multiplications of 2x2x2
    # Level 2: each 2x2x2 uses 7 multiplications of 1x1x1
    # Total: 7 * 7 = 49
    #
    # So rank 49 for 4x4x4 is achievable.

    # For 4x4x1: A (4x4) * B (4x1) = C (4x1)
    # Schoolbook: 4*4*1 = 16 multiplications.
    # Can we do better? 4x4x1 is a special case.
    # Actually, 4x4x1 is just matrix-vector multiplication, rank is at least 4*1=4?
    # No, the tensor rank of 4x4x1 multiplication is 16 (schoolbook is optimal for this).
    # Actually, let me reconsider. The multiplication tensor for 4x4x1 has rank 16.

    # Total rank: 49 + 49 + 16 = 114

    # Let me build the Strassen 4x4x4 algorithm.
    # First, let me define the Strassen 2x2x2 algorithm.
    # For 2x2x2: M = [[m11, m12], [m21, m22]] * [[n11, n12], [n21, n22]] = [[p11, p12], [p21, p22]]
    # Strassen:
    # M1 = (m11 + m22) * (n11 + n22)
    # M2 = (m21 + m22) * n11
    # M3 = m11 * (n12 - n22)
    # M4 = m22 * (n21 - n11)
    # M5 = (m11 + m12) * n22
    # M6 = (m21 - m11) * (n11 + n12)
    # M7 = (m12 - m22) * (n21 + n22)
    # p11 = M1 + M4 - M5 + M7
    # p12 = M3 + M5
    # p21 = M2 + M4
    # p22 = M1 - M2 + M3 + M6

    # Now for 4x4x4, we use a hierarchical approach.
    # Split A, B, C into 2x2 blocks of 2x2 submatrices.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2

    # Using Strassen at the top level:
    # S1 = (A11 + A22) * (B11 + B22)  -> 2x2 matrix product, rank 7
    # S2 = (A21 + A22) * B11          -> 2x2 matrix product, rank 7
    # S3 = A11 * (B12 - B22)          -> 2x2 matrix product, rank 7
    # S4 = A22 * (B21 - B11)          -> 2x2 matrix product, rank 7
    # S5 = (A11 + A12) * B22          -> 2x2 matrix product, rank 7
    # S6 = (A21 - A11) * (B11 + B12)  -> 2x2 matrix product, rank 7
    # S7 = (A12 - A22) * (B21 + B22)  -> 2x2 matrix product, rank 7

    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6

    # Each S_i is a 2x2 matrix product, which uses 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.

    # Now I need to construct the U, V, W matrices for this.
    # This is complex. Let me build it step by step.

    # For the 4x4x4 part, I'll create a helper function.
    # U4: 16 x 49, V4: 16 x 49, W4: 16 x 49
    # For the 4x4x1 part: U1: 16 x 16, V1: 4 x 16, W1: 4 x 16

    # Let me define the Strassen 2x2x2 first.
    # For 2x2x2: U2: 4 x 7, V2: 4 x 7, W2: 4 x 7
    # U2[i*2+j, c] corresponds to A[i,j], V2[i*2+j, c] to B[i,j], W2[i*2+j, c] to C[i,j]

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen algorithm."""
        # 7 multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C00 += M1, C11 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += M2, C11 -= M2
        W[2][1] = 1; W[3][1] = -1
        
        # M3: A00*(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C01 += M3, C11 += M3
        W[1][2] = 1; W[3][2] = 1
        
        # M4: A11*(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C00 += M4, C10 += M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (A00+A01)*B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 -= M5, C01 += M5
        W[0][4] = -1; W[1][4] = 1
        
        # M6: (A10-A00)*(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C11 += M6
        W[3][5] = 1
        
        # M7: (A01-A11)*(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C00 += M7
        W[0][6] = 1
        
        return U, V, W

    # Now build 4x4x4 from Strassen 2x2x2.
    # The 4x4 matrices are split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 = [[a00,a01],[a10,a11]], etc.
    # In our indexing: A[i*m+j] for i,j in 0..3
    # A11 = rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A12 = rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A21 = rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A22 = rows 2,3 cols 2,3 -> indices 10,11,14,15

    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 using hierarchical Strassen."""
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Get the 2x2x2 Strassen matrices
        U2, V2, W2 = strassen_2x2x2()
        
        # Top-level Strassen: 7 sub-problems, each is a 2x2x2 multiplication
        # For each top-level multiplication s (0..6), we have a 2x2x2 sub-problem.
        # The 2x2 blocks are mapped to 4x4 indices as follows:
        # Block (bi, bj) of a 2x2 matrix corresponds to rows bi*2..bi*2+1, cols bj*2..bj*2+1
        # In the 4x4 flat indexing: row r, col c -> index r*4+c
        
        # For each of the 7 top-level Strassen steps:
        # s=0: (A11+A22) * (B11+B22)
        # s=1: (A21+A22) * B11
        # s=2: A11 * (B12-B22)
        # s=3: A22 * (B21-B11)
        # s=4: (A11+A12) * B22
        # s=5: (A21-A11) * (B11+B12)
        # s=6: (A12-A22) * (B21+B22)
        
        # For each step s, the 2x2x2 sub-problem has:
        # U_block: 4x7 matrix where U_block[i*2+j, c] gives the coefficient for A[i,j] in the left operand
        # V_block: 4x7 matrix where V_block[i*2+j, c] gives the coefficient for B[i,j] in the right operand
        # W_block: 4x7 matrix where W_block[i*2+j, c] gives the coefficient for C[i,j] in the output
        
        # The mapping from 2x2 block indices to 4x4 flat indices:
        # Block row bi (0 or 1), block col bj (0 or 1)
        # Within block: local row i (0 or 1), local col j (0 or 1)
        # Global row = bi*2 + i, global col = bj*2 + j
        # Flat index = global_row * 4 + global_col
        
        def block_to_flat(bi, bj, i, j):
            """Convert block (bi,bj) and local (i,j) to flat index."""
            gr = bi * 2 + i
            gc = bj * 2 + j
            return gr * 4 + gc
        
        # Define the top-level Strassen operations
        # For each step s, define:
        # left_blocks: list of (bi, bj, coeff) for A
        # right_blocks: list of (bi, bj, coeff) for B
        # out_blocks: list of (bi, bj, coeff) for C
        
        steps = []
        # s=0: (A11+A22) * (B11+B22) -> C11 += 1, C11 -= ... wait
        # C11 = S0 + S3 - S4 + S6
        # C12 = S2 + S4
        # C21 = S1 + S3
        # C22 = S0 - S1 + S2 + S5
        
        # S0: (A11+A22) * (B11+B22)
        steps.append({
            'left': [(0,0,1), (1,1,1)],  # A11 + A22
            'right': [(0,0,1), (1,1,1)], # B11 + B22
            'out': [(0,0,1), (1,1,1)]    # contributes to C11 and C22? No...
        })
        
        # Actually, let me reconsider. The output of each S_s is a 2x2 matrix.
        # S0 contributes to C11 and C22? No, each S_s is a full 2x2 matrix.
        # C11 = S0 + S3 - S4 + S6 (all 2x2 matrices)
        # C12 = S2 + S4
        # C21 = S1 + S3
        # C22 = S0 - S1 + S2 + S5
        
        # So S0 (a 2x2 matrix) contributes +1 to C11 and +1 to C22? No!
        # S0 is a 2x2 matrix, and it contributes to C11 (the top-left 2x2 block of C)
        # and to C22 (the bottom-right 2x2 block of C)? No, that's not right either.
        
        # Let me re-read Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Each M_i is a 2x2 matrix. So M1 contributes to C11 and C22.
        # M2 contributes to C21 and C22 (with -1 for C22).
        # M3 contributes to C12 and C22.
        # M4 contributes to C11 and C21.
        # M5 contributes to C11 (-1) and C12.
        # M6 contributes to C22.
        # M7 contributes to C11.
        
        # So for the 4x4x4 case:
        # M0 (S0): (A11+A22)*(B11+B22), contributes +1 to C11, +1 to C22
        # M1 (S1): (A21+A22)*B11, contributes +1 to C21, -1 to C22
        # M2 (S2): A11*(B12-B22), contributes +1 to C12, +1 to C22
        # M3 (S3): A22*(B21-B11), contributes +1 to C11, +1 to C21
        # M4 (S4): (A11+A12)*B22, contributes -1 to C11, +1 to C12
        # M5 (S5): (A21-A11)*(B11+B12), contributes +1 to C22
        # M6 (S6): (A12-A22)*(B21+B22), contributes +1 to C11
        
        # Now, for each M_s, the 2x2x2 sub-problem is:
        # Left operand: a 2x2 matrix formed from blocks of A
        # Right operand: a 2x2 matrix formed from blocks of B
        # Output: a 2x2 matrix that gets added to the appropriate C blocks
        
        # For M0: Left = A11 + A22 (2x2 matrix), Right = B11 + B22 (2x2 matrix)
        # The 2x2x2 Strassen algorithm is applied to these 2x2 matrices.
        # But A11 and A22 are themselves 2x2 matrices (in the 4x4 context).
        # So the "2x2 matrix" for the sub-problem has entries that are 2x2 blocks.
        # This means we need to apply the 2x2x2 Strassen at the block level.
        
        # This is getting complex. Let me just directly construct the 49 multiplications.
        
        # Alternative approach: directly enumerate the 49 scalar multiplications.
        # For the hierarchical Strassen, each of the 7 top-level multiplications
        # involves 7 scalar multiplications (from the inner Strassen).
        
        # Let me define the full 49-term decomposition directly.
        
        # I'll use a different approach: build it from the known rank-49 algorithm.
        # The rank-49 algorithm for 4x4x4 is well-known. Let me construct it.
        
        # Actually, let me just use the schoolbook for 4x4x4 (rank 64) and 4x4x1 (rank 16)
        # Total: 64 + 64 + 16 = 144. That's the same as the seed.
        
        # Let me try a different split: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
        # Using Strassen for the 4x4x4 parts: 49 + 49 + 16 = 114
        
        # I need to properly construct the rank-49 4x4x4 algorithm.
        
        # Let me build it carefully.
        
        # The 4x4x4 multiplication: C = A * B
        # A is 4x4, B is 4x4, C is 4x4
        # Split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where Aij is 2x2
        # B = [[B00, B01], [B10, B11]] where Bij is 2x2
        # C = [[C00, C01], [C10, C11]] where Cij is 2x2
        
        # Strassen:
        # M0 = (A00+A11)*(B00+B11)
        # M1 = (A10+A11)*B00
        # M2 = A00*(B01-B11)
        # M3 = A11*(B10-B00)
        # M4 = (A00+A01)*B11
        # M5 = (A10-A00)*(B00+B01)
        # M6 = (A01-A11)*(B10+B11)
        #
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # Each M_s is a 2x2 matrix product. To compute M_s, we use Strassen's 2x2x2.
        # So M_s = L_s * R_s where L_s and R_s are 2x2 matrices.
        # The 2x2x2 Strassen uses 7 scalar multiplications.
        
        # For M0: L0 = A00+A11, R0 = B00+B11
        # L0 is a 2x2 matrix where each entry is a 2x2 block from A.
        # Specifically, L0[i,j] = A00[i,j] + A11[i,j] for i,j in 0,1
        # In the 4x4 flat indexing:
        # A00[i,j] = A[i,j] (top-left 2x2)
        # A11[i,j] = A[i+2,j+2] (bottom-right 2x2)
        
        # This is very complex to construct directly. Let me use a different approach.
        
        # I'll construct the U, V, W matrices by directly computing the tensor.
        
        # For now, let me use a simpler approach: schoolbook for everything.
        # 4x4x9 schoolbook: rank = 4*4*9 = 144
        
        # But I want to do better. Let me try to implement the Strassen 4x4x4.
        
        # I'll define the 49 multiplications explicitly.
        
        # For each of the 7 top-level Strassen steps, and for each of the 7 inner Strassen steps,
        # I have a scalar multiplication.
        
        # Let me define the mapping.
        
        # Top-level step s (0..6):
        #   Left operand L_s: 2x2 matrix, each entry is a linear combination of 2x2 blocks of A
        #   Right operand R_s: 2x2 matrix, each entry is a linear combination of 2x2 blocks of B
        #   Output: contributes to C blocks with certain signs
        
        # Inner Strassen step t (0..6):
        #   Computes one scalar multiplication for the 2x2x2 problem
        
        # For the 2x2x2 Strassen:
        #   Inner step 0: (L00+L11)*(R00+R11)
        #   Inner step 1: (L10+L11)*R00
        #   Inner step 2: L00*(R01-R11)
        #   Inner step 3: L11*(R10-R00)
        #   Inner step 4: (L00+L01)*R11
        #   Inner step 5: (L10-L00)*(R00+R01)
        #   Inner step 6: (L01-L11)*(R10+R11)
        
        # Where Lij and Rij are the entries of the 2x2 matrices L and R.
        
        # Now, Lij is a linear combination of 2x2 blocks of A.
        # Each 2x2 block of A has 4 scalar entries.
        # So Lij is a 2x2 matrix of scalars (in the 4x4 context).
        
        # Wait, I'm confusing myself. Let me restart.
        
        # In the 4x4x4 problem:
        # A is a 4x4 matrix of scalars.
        # We split A into 2x2 blocks: A00, A01, A10, A11, each 2x2.
        # The "2x2 matrix" L = A00 + A11 is a 2x2 matrix where each entry is a scalar.
        # No wait, A00 is a 2x2 matrix of scalars, so L = A00 + A11 is also a 2x2 matrix of scalars.
        # So L is just a 2x2 matrix, and we can apply the 2x2x2 Strassen directly.
        
        # So the hierarchical Strassen for 4x4x4 is:
        # 1. Form the 7 linear combinations of 2x2 blocks.
        # 2. For each pair, apply 2x2x2 Strassen (7 scalar multiplications).
        # 3. Combine the results.
        
        # Total: 7 * 7 = 49 scalar multiplications.
        
        # Now let me construct the U, V, W matrices.
        
        # I'll index the 49 multiplications as (s, t) where s is the top-level step and t is the inner step.
        # Column index: c = s * 7 + t
        
        # For each (s, t), I need to determine:
        # - Which entries of A are involved (and their coefficients) -> U
        # - Which entries of B are involved (and their coefficients) -> V
        # - Which entries of C are involved (and their coefficients) -> W
        
        # Let me define the top-level steps:
        # s=0: L = A00+A11, R = B00+B11, out: C00 += 1, C11 += 1
        # s=1: L = A10+A11, R = B00, out: C10 += 1, C11 -= 1
        # s=2: L = A00, R = B01-B11, out: C01 += 1, C11 += 1
        # s=3: L = A11, R = B10-B00, out: C00 += 1, C10 += 1
        # s=4: L = A00+A01, R = B11, out: C00 -= 1, C01 += 1
        # s=5: L = A10-A00, R = B00+B01, out: C11 += 1
        # s=6: L = A01-A11, R = B10+B11, out: C00 += 1
        
        # For the inner Strassen (2x2x2):
        # t=0: (L00+L11)*(R00+R11)
        # t=1: (L10+L11)*R00
        # t=2: L00*(R01-R11)
        # t=3: L11*(R10-R00)
        # t=4: (L00+L01)*R11
        # t=5: (L10-L00)*(R00+R01)
        # t=6: (L01-L11)*(R10+R11)
        
        # Where Lij and Rij are the (i,j) entries of the 2x2 matrices L and R.
        
        # Now, Lij is a linear combination of entries of A.
        # For example, for s=0: L = A00 + A11
        # L00 = A[0,0] + A[2,2]
        # L01 = A[0,1] + A[2,3]
        # L10 = A[1,0] + A[3,2]
        # L11 = A[1,1] + A[3,3]
        
        # In flat indexing (row-major, 4x4):
        # A[i,j] -> index i*4+j
        # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
        # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
        # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
        # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
        
        # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = [[0,1],[4,5]]
        # A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]] = [[2,3],[6,7]]
        # A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]] = [[8,9],[12,13]]
        # A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]] = [[10,11],[14,15]]
        
        # Similarly for B and C.
        
        # This is very tedious to do by hand. Let me write code to generate it.
        
        # I'll create a function that, given the top-level step s, returns:
        # - left_coeffs: a 2x2 matrix where each entry is a list of (flat_index, coeff) for A
        # - right_coeffs: a 2x2 matrix where each entry is a list of (flat_index, coeff) for B
        # - out_blocks: a list of (block_row, block_col, coeff) for C
        
        # Then for each inner step t, I'll determine which Lij and Rij are involved.
        
        # Let me code this up.
        
        def get_top_step(s):
            """Return left_coeffs, right_coeffs, out_blocks for top-level step s."""
            # A blocks in flat indexing:
            # A00: [0,1,4,5], A01: [2,3,6,7], A10: [8,9,12,13], A11: [10,11,14,15]
            # B blocks: same structure
            # C blocks: same structure
            
            A00 = [[(0,1),(1,1)],[(4,1),(5,1)]]
            A01 = [[(2,1),(3,1)],[(6,1),(7,1)]]
            A10 = [[(8,1),(9,1)],[(12,1),(13,1)]]
            A11 = [[(10,1),(11,1)],[(14,1),(15,1)]]
            
            B00 = [[(0,1),(1,1)],[(4,1),(5,1)]]
            B01 = [[(2,1),(3,1)],[(6,1),(7,1)]]
            B10 = [[(8,1),(9,1)],[(12,1),(13,1)]]
            B11 = [[(10,1),(11,1)],[(14,1),(15,1)]]
            
            if s == 0:
                # L = A00+A11, R = B00+B11
                left = [[(A00[i][j][0], A00[i][j][1]+A11[i][j][1]) for j in range(2)] for i in range(2)]
                right = [[(B00[i][j][0], B00[i][j][1]+B11[i][j][1]) for j in range(2)] for i in range(2)]
                out = [(0,0,1),(1,1,1)]
            elif s == 1:
                # L = A10+A11, R = B00
                left = [[(A10[i][j][0], A10[i][j][1]+A11[i][j][1]) for j in range(2)] for i in range(2)]
                right = B00
                out = [(1,0,1),(1,1,-1)]
            elif s == 2:
                # L = A00, R = B01-B11
                left = A00
                right = [[(B01[i][j][0], B01[i][j][1]-B11[i][j][1]) for j in range(2)] for i in range(2)]
                out = [(0,1,1),(1,1,1)]
            elif s == 3:
                # L = A11, R = B10-B00
                left = A11
                right = [[(B10[i][j][0], B10[i][j][1]-B00[i][j][1]) for j in range(2)] for i in range(2)]
                out = [(0,0,1),(1,0,1)]
            elif s == 4:
                # L = A00+A01, R = B11
                left = [[(A00[i][j][0], A00[i][j][1]+A01[i][j][1]) for j in range(2)] for i in range(2)]
                right = B11
                out = [(0,0,-1),(0,1,1)]
            elif s == 5:
                # L = A10-A00, R = B00+B01
                left = [[(A10[i][j][0], A10[i][j][1]-A00[i][j][1]) for j in range(2)] for i in range(2)]
                right = [[(B00[i][j][0], B00[i][j][1]+B01[i][j][1]) for j in range(2)] for i in range(2)]
                out = [(1,1,1)]
            elif s == 6:
                # L = A01-A11, R = B10+B11
                left = [[(A01[i][j][0], A01[i][j][1]-A11[i][j][1]) for j in range(2)] for i in range(2)]
                right = [[(B10[i][j][0], B10[i][j][1]+B11[i][j][1]) for j in range(2)] for i in range(2)]
                out = [(0,0,1)]
            else:
                raise ValueError(f"Invalid step {s}")
            
            return left, right, out
        
        # Now build the 4x4x4 U, V, W.
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # Inner Strassen steps: for each t, determine which Lij and Rij are involved.
        # t=0: (L00+L11)*(R00+R11)
        # t=1: (L10+L11)*R00
        # t=2: L00*(R01-R11)
        # t=3: L11*(R10-R00)
        # t=4: (L00+L01)*R11
        # t=5: (L10-L00)*(R00+R01)
        # t=6: (L01-L11)*(R10+R11)
        
        # For each t, the left operand involves certain Lij with coefficients,
        # and the right operand involves certain Rij with coefficients.
        
        inner_steps = [
            # (left_L_indices, left_coeffs, right_R_indices, right_coeffs)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # t=0
            ([(1,0,1),(1,1,1)], [(0,0,1)]),           # t=1
            ([(0,0,1)], [(0,1,1),(1,1,-1)]),          # t=2
            ([(1,1,1)], [(1,0,1),(0,0,-1)]),          # t=3
            ([(0,0,1),(0,1,1)], [(1,1,1)]),           # t=4
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),  # t=5
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),  # t=6
        ]
        
        for s in range(7):
            left, right, out = get_top_step(s)
            for t in range(7):
                c = s * 7 + t
                L_indices, R_indices = inner_steps[t]
                
                # Build U contribution
                for (li, lj, lc) in L_indices:
                    for (flat_idx, coeff) in left[li][lj]:
                        U4[flat_idx][c] += lc * coeff
                
                # Build V contribution
                for (ri, rj, rc) in R_indices:
                    for (flat_idx, coeff) in right[ri][rj]:
                        V4[flat_idx][c] += rc * coeff
                
                # Build W contribution
                for (bi, bj, bc) in out:
                    # The output 2x2 matrix from the inner Strassen contributes to C block (bi,bj)
                    # The inner Strassen output is a 2x2 matrix, so we need to map it to C.
                    # The inner Strassen computes C_inner = L_inner * R_inner (2x2x2)
                    # C_inner[i,j] = sum over the 7 terms
                    # For the 2x2x2 Strassen:
                    # C00 = M0 + M3 - M4 + M6
                    # C01 = M2 + M4
                    # C10 = M1 + M3
                    # C11 = M0 - M1 + M2 + M5
                    #
                    # So for inner step t, the contribution to C_inner[i,j] is:
                    inner_W = {
                        0: [(0,0,1),(1,1,1)],
                        1: [(1,0,1),(1,1,-1)],
                        2: [(0,1,1),(1,1,1)],
                        3: [(0,0,1),(1,0,1)],
                        4: [(0,0,-1),(0,1,1)],
                        5: [(1,1,1)],
                        6: [(0,0,1)],
                    }
                    for (ci, cj, cc) in inner_W[t]:
                        # C_inner[ci,cj] contributes to C block (bi,bj) at position (ci,cj)
                        # Global C index: block (bi,bj), local (ci,cj)
                        # Global row = bi*2+ci, global col = bj*2+cj
                        gr = bi * 2 + ci
                        gc = bj * 2 + cj
                        flat_c = gr * 4 + gc
                        W4[flat_c][c] += bc * cc
        
        # Now build the 4x4x1 part.
        # A (4x4) * B (4x1) = C (4x1)
        # Schoolbook: C[i] = sum_j A[i,j] * B[j]
        # 16 multiplications.
        
        U1 = [[0]*16 for _ in range(16)]
        V1 = [[0]*16 for _ in range(4)]
        W1 = [[0]*16 for _ in range(4)]
        
        for i in range(4):
            for j in range(4):
                c = i * 4 + j
                U1[i*4+j][c] = 1
                V1[j][c] = 1
                W1[i][c] = 1
        
        # Combine: total rank = 49 + 49 + 16 = 114
        # U: 16 x 114
        # V: 36 x 114 (16 for first 4x4x4, 16 for second 4x4x4, 4 for 4x4x1)
        # W: 36 x 114 (16 for first 4x4x4, 16 for second 4x4x4, 4 for 4x4x1)
        
        total_rank = 49 + 49 + 16
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(36)]
        W = [[0]*total_rank for _ in range(36)]
        
        # First 4x4x4: columns 0-48
        for i in range(16):
            for c in range(49):
                U[i][c] = U4[i][c]
                V[i][c] = V4[i][c]
                W[i][c] = W4[i][c]
        
        # Second 4x4x4: columns 49-97
        # This handles B columns 4-7 (the second 4 columns of the 9-column B)
        # A is the same 4x4, B is columns 4-7, C is columns 4-7
        for i in range(16):
            for c in range(49):
                U[i][49+c] = U4[i][c]
                V[i+16][49+c] = V4[i][c]  # B rows 4-7 (flat index 16-31)
                W[i+16][49+c] = W4[i][c]  # C rows 4-7 (flat index 16-31)
        
        # 4x4x1: columns 98-113
        # A is 4x4, B is column 8 (flat index 32), C is column 8 (flat index 32)
        for i in range(16):
            for c in range(16):
                U[i][98+c] = U1[i][c]
                V[32][98+c] = V1[i][c]  # B row 32 (column 8)
                W[32][98+c] = W1[i][c]  # C row 32 (column 8)
        
        return {"U": U, "V": V, "W": W}

BET: {mechanism: "Using Strassen-based rank-49 for two 4x4x4 subproblems plus rank-16 schoolbook for 4x4x1 gives rank 114, a structural improvement over the seed's rank-144 schoolbook", q: 0.85, delta: [0.2, 0.35]}
