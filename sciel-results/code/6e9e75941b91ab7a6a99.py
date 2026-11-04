import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen's algorithm recursively to achieve rank 49 for 4x4x4 blocks,
    and standard decomposition for the remaining 4x4x1 block.
    
    Total rank: 2 * 49 (for two 4x4x4) + 16 (for 4x4x1) = 114
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll split B (4x9) into B1 (4x4) and B2 (4x5)
    # B2 can be split into B2a (4x4) and B2b (4x1)
    # So: C = A*B1 + A*B2a + A*B2b
    # First two terms are 4x4x4 multiplications (Strassen rank 49 each)
    # Last term is 4x4x1 multiplication (standard rank 16)
    
    rank_4x4x4 = 49  # Strassen
    rank_4x4x1 = 16  # Standard
    total_rank = 2 * rank_4x4x4 + rank_4x4x1
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    def build_strassen_2x2():
        """Build Strassen decomposition for 2x2 x 2x2 -> rank 7"""
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[1, 3] = 1
        
        # M5 = (A11+A12)B22
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2
    
    def build_strassen_4x4x4():
        """
        Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
        Uses 7 Strassen combinations at the 4x4 level, each being a 2x2x2 
        multiplication done via Strassen (rank 7).
        """
        U2, V2, W2 = build_strassen_2x2()
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # The 7 Strassen combinations at 4x4 level:
        # Each involves sums/differences of 2x2 blocks
        # For each combination, we need to map to the 2x2x2 Strassen
        
        # Block indexing: A block (bi, bk), B block (bk, bj), C block (bj, bi)
        # where bi, bk, bj in {0, 1}
        
        # Strassen combinations:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For each combination, we have a 2x2 matrix of 2x2 blocks
        # We apply Strassen 2x2x2 to each
        
        # Let's define the block coefficients for each Strassen term
        # For term t, we have A_blocks[t] = {(bi, bk): coeff}
        # B_blocks[t] = {(bk, bj): coeff}
        # C_blocks[t] = {(bj, bi): coeff}
        
        strassen_terms = [
            # M1: (A11+A22)(B11+B22)
            {
                'A': {((0,0), (0,0)): 1, ((1,1), (1,1)): 1},
                'B': {((0,0), (0,0)): 1, ((1,1), (1,1)): 1},
                'C': {((0,0), (0,0)): 1, ((1,1), (1,1)): 1}
            },
            # M2: (A21+A22)B11
            {
                'A': {((1,0), (0,0)): 1, ((1,1), (1,1)): 1},
                'B': {((0,0), (0,0)): 1},
                'C': {((0,1), (0,0)): 1, ((1,1), (1,1)): -1}
            },
            # M3: A11(B12-B22)
            {
                'A': {((0,0), (0,0)): 1},
                'B': {((0,1), (0,0)): 1, ((1,1), (1,1)): -1},
                'C': {((1,0), (0,0)): 1, ((1,1), (1,1)): 1}
            },
            # M4: A22(B21-B11)
            {
                'A': {((1,1), (1,1)): 1},
                'B': {((1,0), (0,0)): 1, ((0,0), (0,0)): -1},
                'C': {((0,0), (0,0)): 1, ((0,1), (1,1)): 1}
            },
            # M5: (A11+A12)B22
            {
                'A': {((0,0), (0,0)): 1, ((0,1), (0,0)): 1},
                'B': {((1,1), (1,1)): 1},
                'C': {((0,0), (0,0)): -1, ((1,0), (0,0)): 1}
            },
            # M6: (A21-A11)(B11+B12)
            {
                'A': {((1,0), (0,0)): 1, ((0,0), (0,0)): -1},
                'B': {((0,0), (0,0)): 1, ((0,1), (0,0)): 1},
                'C': {((1,1), (1,1)): 1}
            },
            # M7: (A12-A22)(B21+B22)
            {
                'A': {((0,1), (0,0)): 1, ((1,1), (1,1)): -1},
                'B': {((1,0), (0,0)): 1, ((1,1), (1,1)): 1},
                'C': {((0,0), (0,0)): 1}
            }
        ]
        
        # This approach is getting complex. Let me use a cleaner recursive approach.
        # Actually, let me just directly construct the 4x4x4 Strassen using the known rank-49.
        
        # Alternative: Use the fact that 4x4x4 can be done with rank 49 by 
        # applying Strassen at the block level where blocks are 2x2.
        
        # Let me restart with a clearer construction.
        
        # For 4x4x4, we have 7 big multiplications, each of which is a 2x2x2 
        # matrix multiplication (of 2x2 blocks). Each 2x2x2 has rank 7 via Strassen.
        # So total rank = 7 * 7 = 49.
        
        # The 7 big multiplications are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Where A11, A12, A21, A22 are 2x2 blocks of A, etc.
        
        # For each Mi, we need to compute a 2x2 x 2x2 multiplication.
        # The input to this 2x2x2 multiplication is formed by combining blocks.
        
        # Let me define for each Mi:
        # - Which A blocks contribute (with signs) to form the "A matrix" of the 2x2x2
        # - Which B blocks contribute (with signs) to form the "B matrix" of the 2x2x2
        # - Which C blocks receive the result (with signs)
        
        # Actually, each Mi IS a 2x2x2 multiplication. The "A" for Mi is a 2x2 matrix 
        # formed from A blocks, "B" is a 2x2 matrix formed from B blocks, and the result 
        # is added to C blocks.
        
        # Let me define this more carefully.
        
        # For M1 = (A11+A22)(B11+B22):
        # The "A matrix" for the 2x2x2 is [[A11, 0], [0, A22]] + [[0,0],[0,0]]... 
        # No, that's not right.
        
        # Actually, M1 = (A11+A22)(B11+B22) means we're multiplying two 2x2 matrices:
        # A_M1 = A11 + A22 (but these are 2x2 blocks, so A_M1 is a 2x2 matrix where 
        #         A_M1[0,0] = A11[0,0]+A22[0,0], etc.)
        # B_M1 = B11 + B22 (similarly)
        # Result: C11 += M1, C22 += M1
        
        # So for the 2x2x2 Strassen, we need to express A_M1 and B_M1 in terms of 
        # the original 4x4 entries.
        
        # This is getting very complex. Let me use a different approach.
        
        # I'll use the known fact that the rank of 4x4x4 matrix multiplication is 49 
        # (achieved by Strassen), and construct it by embedding the 2x2x2 Strassen 
        # into a 4x4x4 structure.
        
        # Actually, let me just return the schoolbook for now to ensure correctness,
        # but with a better structure.
        
        # For the combined program, let me use a hybrid approach:
        # - Use Strassen for the 4x4x4 parts (rank 49 each)
        # - Use standard for 4x4x1 (rank 16)
        # - Total: 114
        
        # But I need to actually construct the Strassen 4x4x4 correctly.
        
        # Let me use a recursive construction.
        
        return None  # Placeholder, will implement below
    
    # Let me implement the full construction properly.
    
    def build_strassen_2x2():
        """Build Strassen decomposition for 2x2 x 2x2 -> rank 7"""
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[1, 3] = 1
        
        # M5 = (A11+A12)B22
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2
    
    def build_4x4x4_strassen():
        """
        Build rank-49 decomposition for 4x4 x 4x4 using Strassen.
        """
        U2, V2, W2 = build_strassen_2x2()
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # The 7 Strassen combinations at the 4x4 level
        # Each is a 2x2 x 2x2 multiplication of 2x2 blocks
        
        # For each combination, define:
        # a_blocks: dict mapping (block_row, block_col) -> coefficient for A
        # b_blocks: dict mapping (block_row, block_col) -> coefficient for B  
        # c_blocks: dict mapping (block_row, block_col) -> coefficient for C
        
        # Block (r, c) of a 4x4 matrix covers rows 2r to 2r+1, cols 2c to 2c+1
        
        strassen_defs = [
            # M1 = (A11+A22)(B11+B22)
            {
                'a': {((0,0), (0,0)): 1, ((1,1), (1,1)): 1},  # A11 + A22
                'b': {((0,0), (0,0)): 1, ((1,1), (1,1)): 1},  # B11 + B22
                'c': {((0,0), (0,0)): 1, ((1,1), (1,1)): 1}   # C11 += M1, C22 += M1
            },
            # M2 = (A21+A22)B11
            {
                'a': {((1,0), (0,0)): 1, ((1,1), (1,1)): 1},  # A21 + A22
                'b': {((0,0), (0,0)): 1},                      # B11
                'c': {((0,1), (0,0)): 1, ((1,1), (1,1)): -1}   # C21 += M2, C22 -= M2
            },
            # M3 = A11(B12-B22)
            {
                'a': {((0,0), (0,0)): 1},                      # A11
                'b': {((0,1), (0,0)): 1, ((1,1), (1,1)): -1},  # B12 - B22
                'c': {((1,0), (0,0)): 1, ((1,1), (1,1)): 1}    # C12 += M3, C22 += M3
            },
            # M4 = A22(B21-B11)
            {
                'a': {((1,1), (1,1)): 1},                      # A22
                'b': {((1,0), (0,0)): 1, ((0,0), (0,0)): -1},  # B21 - B11
                'c': {((0,0), (0,0)): 1, ((0,1), (1,1)): 1}    # C11 += M4, C21 += M4
            },
            # M5 = (A11+A12)B22
            {
                'a': {((0,0), (0,0)): 1, ((0,1), (0,0)): 1},   # A11 + A12
                'b': {((1,1), (1,1)): 1},                      # B22
                'c': {((0,0), (0,0)): -1, ((1,0), (0,0)): 1}   # C11 -= M5, C12 += M5
            },
            # M6 = (A21-A11)(B11+B12)
            {
                'a': {((1,0), (0,0)): 1, ((0,0), (0,0)): -1},  # A21 - A11
                'b': {((0,0), (0,0)): 1, ((0,1), (0,0)): 1},   # B11 + B12
                'c': {((1,1), (1,1)): 1}                       # C22 += M6
            },
            # M7 = (A12-A22)(B21+B22)
            {
                'a': {((0,1), (0,0)): 1, ((1,1), (1,1)): -1},  # A12 - A22
                'b': {((1,0), (0,0)): 1, ((1,1), (1,1)): 1},   # B21 + B22
                'c': {((0,0), (0,0)): 1}                       # C11 += M7
            }
        ]
        
        # For each Strassen combination, we perform a 2x2x2 multiplication
        # The 2x2x2 uses Strassen (rank 7)
        
        # For combination t, the "A matrix" for the 2x2x2 is formed by:
        # A_2x2[i,j] = sum of coeff * A_block[bi,bk] where (bi,bk) maps to (i,j)
        # Similarly for B and C.
        
        # Actually, this is tricky because the blocks are 2x2, so we need to 
        # map the 2x2 block structure to the 4x4 indices.
        
        # Let me think about this differently.
        
        # For each Strassen term t:
        # We have a 2x2 "meta-matrix" A_t where A_t[i,j] is a 2x2 block
        # A_t[i,j] = sum_{(bi,bk) in a_blocks} coeff * A[2*bi:2*bi+2, 2*bk:2*bk+2]
        
        # Similarly for B_t and C_t.
        
        # Then C_t = A_t * B_t (2x2 matrix multiplication of 2x2 blocks)
        # And C += C_t
        
        # The 2x2 matrix multiplication A_t * B_t can be done with Strassen (rank 7).
        
        # So for each t, we have 7 "sub-multiplications" (the Strassen terms for the 2x2x2).
        
        # Total: 7 * 7 = 49 multiplications.
        
        for t, sdef in enumerate(strassen_defs):
            base_col = t * 7
            
            # For each of the 7 Strassen sub-terms in the 2x2x2:
            for sub in range(7):
                col = base_col + sub
                
                # Get the Strassen 2x2x2 coefficients for this sub-term
                # U2[i, sub], V2[k, sub], W2[j, sub]
                
                # U2 is indexed by (i, k) where i is row of A, k is col of A (which is row of B)
                # V2 is indexed by (k, j) where k is row of B, j is col of B (which is row of C)
                # W2 is indexed by (j, i) where j is row of C, i is col of C (which is row of A)
                
                # For the 2x2x2 Strassen sub-term 'sub':
                # The A part involves A_2x2[i, k] for certain (i, k)
                # The B part involves B_2x2[k, j] for certain (k, j)
                # The C part involves C_2x2[j, i] for certain (j, i)
                
                # A_2x2[i, k] = sum_{(bi,bk) in a_blocks} coeff * A_block[bi,bk]
                # where A_block[bi,bk] is the 2x2 block at position (bi, bk)
                
                # So for U2[i, sub] != 0, we need to add to U4 the contributions from 
                # all A blocks that contribute to A_2x2[i, k]
                
                # Let me extract the non-zero entries of U2, V2, W2 for this sub-term
                
                u_entries = [(i, k) for i in range(2) for k in range(2) if U2[i*2+k, sub] != 0]
                v_entries = [(k, j) for k in range(2) for j in range(2) if V2[k*2+j, sub] != 0]
                w_entries = [(j, i) for j in range(2) for i in range(2) if W2[j*2+i, sub] != 0]
                
                # For each (i, k) in u_entries with coefficient u_coeff:
                # A_2x2[i, k] contributes u_coeff to the U part
                # A_2x2[i, k] = sum_{(bi,bk) in a_blocks} a_coeff * A_block[bi,bk]
                # So we add a_coeff * u_coeff to U4 for each entry in A_block[bi,bk]
                
                for (i, k), u_coeff in [(e, U2[e[0]*2+e[1], sub]) for e in u_entries]:
                    for (bi, bk), a_coeff in sdef['a'].items():
                        # A_block[bi,bk] is at rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
                        # In U4, the index is i*m + k where m=4, so row = i*4 + k
                        # But wait, U4 is for 4x4x4, so U4[i*4+k, col]
                        
                        # For each entry in the 2x2 block:
                        for di in range(2):
                            for dk in range(2):
                                row = (i*2 + di) * 4 + (k*2 + dk)  # This is wrong
                                # Actually, A is 4x4, indexed by (row, col)
                                # A_block[bi,bk] has entries at (2*bi+di, 2*bk+dk)
                                # In U4, the index is (row, col) = (2*bi+di, 2*bk+dk)
                                # So U4 index = (2*bi+di) * 4 + (2*bk+dk)
                                
                                row_idx = (2*bi + di) * 4 + (2*bk + dk)
                                U4[row_idx, col] += a_coeff * u_coeff
                
                # Similarly for V2
                for (k, j), v_coeff in [(e, V2[e[0]*2+e[1], sub]) for e in v_entries]:
                    for (bk, bj), b_coeff in sdef['b'].items():
                        for dk in range(2):
                            for dj in range(2):
                                # B_block[bk,bj] has entries at (2*bk+dk, 2*bj+dj)
                                # In V4, index = (2*bk+dk) * 9 + (2*bj+dj) but wait, V is for m*p = 4*9
                                # Actually for 4x4x4, V is 4x4, so V4[k*4+j, col]
                                row_idx = (2*bk + dk) * 4 + (2*bj + dj)
                                V4[row_idx, col] += b_coeff * v_coeff
                
                # Similarly for W2
                for (j, i), w_coeff in [(e, W2[e[0]*2+e[1], sub]) for e in w_entries]:
                    for (bj, bi), c_coeff in sdef['c'].items():
                        for dj in range(2):
                            for di in range(2):
                                # C_block[bj,bi] has entries at (2*bj+dj, 2*bi+di)
                                # In W4, index = (2*bj+dj) * 4 + (2*bi+di)
                                row_idx = (2*bj + dj) * 4 + (2*bi + di)
                                W4[row_idx, col] += c_coeff * w_coeff
        
        return U4, V4, W4
    
    # Build the 4x4x4 Strassen decomposition
    U4, V4, W4 = build_4x4x4_strassen()
    
    # Now build the full 4x4x9 decomposition
    # C = A * B where A is 4x4, B is 4x9
    # Split B into B1 (4x4) and B2 (4x5)
    # B2 = B2a (4x4) + B2b (4x1)
    
    # C = A*B1 + A*B2a + A*B2b
    
    # First term: A*B1, both 4x4 -> use Strassen rank 49
    # Second term: A*B2a, both 4x4 -> use Strassen rank 49
    # Third term: A*B2b, A is 4x4, B2b is 4x1 -> standard rank 16
    
    # For the first term (A*B1):
    # U4 contributes to U, V4 contributes to V (for cols 0-3 of B), W4 contributes to W (for cols 0-3 of C)
    
    # For the second term (A*B2a):
    # Same U4, but V4 contributes to V (for cols 4-7 of B), W4 contributes to W (for cols 4-7 of C)
    
    # For the third term (A*B2b):
    # Standard 4x4x1: U is 4x4, V is 4x1, W is 1x4
    # Rank 16
    
    # Let me construct this properly.
    
    # Total rank = 49 + 49 + 16 = 114
    
    U_full = np.zeros((16, 114))
    V_full = np.zeros((36, 114))  # 4*9 = 36
    W_full = np.zeros((36, 114))  # 9*4 = 36
    
    # Term 1: A*B1 (B cols 0-3, C cols 0-3)
    U_full[:, 0:49] = U4
    # V4 is 16x49, but V_full is 36x114
    # V4[k*4+j, col] corresponds to V_full[k*9+j, col] for j in 0-3
    for k in range(4):
        for j in range(4):
            V_full[k*9+j, 0:49] = V4[k*4+j, :]
    # W4 is 16x49, W_full is 36x114
    # W4[j*4+i, col] corresponds to W_full[j*4+i, col] for j in 0-3
    for j in range(4):
        for i in range(4):
            W_full[j*4+i, 0:49] = W4[j*4+i, :]
    
    # Term 2: A*B2a (B cols 4-7, C cols 4-7)
    U_full[:, 49:98] = U4
    for k in range(4):
        for j in range(4):
            V_full[k*9+(j+4), 49:98] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W_full[(j+4)*4+i, 49:98] = W4[j*4+i, :]
    
    # Term 3: A*B2b (B cols 8, C col 8)
    # Standard 4x4x1
    for i in range(4):
        for k in range(4):
            col = 98 + i*4 + k
            U_full[i*4+k, col] = 1
            V_full[k*9+8, col] = 1
            W_full[8*4+i, col] = 1
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
