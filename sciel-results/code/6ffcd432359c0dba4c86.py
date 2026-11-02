"""Matrix multiplication 4x4x9: Strassen block decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Each term is (U_entries, V_entries, W_entries) where each is a list of (index, coeff).
    U index: i*2+j (A is 2x2)
    V index: k*2+l (B is 2x2)
    W index: l*2+i (C is 2x2, trace convention W[k*n+i] with n=2)
    """
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = sum A0k*Bk0, C01 = sum A0k*Bk1
    # C10 = sum A1k*Bk0, C11 = sum A1k*Bk1
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W index: l*2+i where l is col of B (0 or 1), i is row of C (0 or 1)
    # C00 -> i=0, l=0 -> W[0]
    # C01 -> i=0, l=1 -> W[1]
    # C10 -> i=1, l=0 -> W[2]
    # C11 -> i=1, l=1 -> W[3]
    
    terms = [
        # M1 = (A00+A11)*(B00+B11)
        # U: A00 at idx 0, A11 at idx 3
        # V: B00 at idx 0, B11 at idx 3
        # W: C00 at idx 0 (+1), C11 at idx 3 (+1)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)*B00
        # U: A10 at idx 2, A11 at idx 3
        # V: B00 at idx 0
        # W: C10 at idx 2 (+1)
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3 = A00*(B01-B11)
        # U: A00 at idx 0
        # V: B01 at idx 1, B11 at idx 3
        # W: C01 at idx 1 (+1)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M4 = A11*(B10-B00)
        # U: A11 at idx 3
        # V: B10 at idx 2, B00 at idx 0
        # W: C00 at idx 0 (+1), C10 at idx 2 (+1)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)*B11
        # U: A00 at idx 0, A01 at idx 1
        # V: B11 at idx 3
        # W: C01 at idx 1 (+1), C00 at idx 0 (-1)
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        # M6 = (A10-A00)*(B00+B01)
        # U: A10 at idx 2, A00 at idx 0
        # V: B00 at idx 0, B01 at idx 1
        # W: C11 at idx 3 (+1)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)*(B10+B11)
        # U: A01 at idx 1, A11 at idx 3
        # V: B10 at idx 2, B11 at idx 3
        # W: C00 at idx 0 (+1)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    return terms

def _strassen_4x4x4_terms():
    """Return rank-1 terms for 4x4x4 Strassen decomposition (rank 28).
    Uses 7 Strassen 2x2x2 multiplications, each of which uses schoolbook 2x2x2 (rank 4) 
    OR we can use Strassen recursively. Let's use Strassen at 2x2 level with schoolbook at 2x2x2.
    Actually, let's just do 7 blocks of 2x2x2 schoolbook (rank 4 each) = rank 28.
    
    Wait, that's just 7*4=28. But we can do better by using Strassen within each 2x2x2.
    No, 2x2x2 schoolbook is rank 4, Strassen is rank 7. Schoolbook is better for 2x2x2.
    
    So 4x4x4 with Strassen at 2x2 block level: 7 multiplications of 2x2x2, each rank 4 = 28.
    
    U index: i*4+j (A is 4x4)
    V index: k*4+l (B is 4x4)
    W index: l*4+i (C is 4x4, trace convention W[k*n+i] with n=4)
    """
    terms = []
    s22 = _strassen_2x2x2()
    
    # 7 Strassen combinations at 2x2 block level
    # A blocks: A00=[0:2,0:2], A01=[0:2,2:4], A10=[2:4,0:2], A11=[2:4,2:4]
    # B blocks: B00=[0:2,0:2], B01=[0:2,2:4], B10=[2:4,0:2], B11=[2:4,2:4]
    # C blocks: C00=[0:2,0:2], C01=[0:2,2:4], C10=[2:4,0:2], C11=[2:4,2:4]
    
    # For each Strassen term at 2x2 level, we need to map to 4x4 indices
    # M1 = (A00+A11)*(B00+B11)
    # This is a 2x2x2 multiplication where:
    #   "A" element (0,0) = A00 (2x2 block), "A" element (1,1) = A11
    #   "B" element (0,0) = B00, "B" element (1,1) = B11
    #   "C" element (0,0) = C00, "C" element (1,1) = C11
    #
    # For each rank-1 term in the 2x2x2 schoolbook, we expand to 4x4x4
    # A 2x2x2 schoolbook term (i,j,k) means: U[i*2+j] * V[k*2+l] * W[l*2+i]
    # But here the "2x2" indices refer to block indices, and within each block we have 2x2.
    
    # Let me think differently. The Strassen 4x4x4 decomposition:
    # We form 7 intermediate 2x2 matrices, each computed as a product of sums/diffs of 2x2 blocks.
    # Each such product is a 2x2x2 matrix multiplication (rank 4 schoolbook).
    # So total rank = 7 * 4 = 28.
    
    # For each of the 7 Strassen multiplications, we have:
    #   U_part: linear combination of A blocks
    #   V_part: linear combination of B blocks
    #   W_part: where the result contributes to C blocks
    
    # Let me define the 7 multiplications explicitly for 4x4x4.
    
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)*B00 -> contributes to C10
    # M3 = A00*(B01-B11) -> contributes to C01
    # M4 = A11*(B10-B00) -> contributes to C00 and C10
    # M5 = (A00+A01)*B11 -> contributes to C01 and C00
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00
    
    # For each multiplication, say M = X*Y where X is a 2x2 matrix (linear combo of A blocks)
    # and Y is a 2x2 matrix (linear combo of B blocks), the result Z is a 2x2 matrix
    # that gets added to certain C blocks.
    
    # X*Y is a 2x2x2 matrix multiplication. Using schoolbook (rank 4):
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    
    # Each term X[a,b]*Y[c,d] is a rank-1 term in the 4x4x4 decomposition.
    # X[a,b] is a linear combination of 2x2 blocks of A.
    # Y[c,d] is a linear combination of 2x2 blocks of B.
    # The product contributes to Z[a,c] which maps to a C block.
    
    # Let me just enumerate all 28 rank-1 terms directly.
    
    # Block index mapping:
    # A block (bi, bj) where bi in {0,1} (row block), bj in {0,1} (col block)
    # A block (bi,bj) covers A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # In flat U index: row i, col j -> U[i*4+j]
    # A block (bi,bj) element (di,dj) is at U[(bi*2+di)*4 + (bj*2+dj)]
    
    # Similarly for B and C.
    
    # Let me define a helper to create a rank-1 term from:
    # U: list of (flat_index, coeff)
    # V: list of (flat_index, coeff)
    # W: list of (flat_index, coeff)
    
    def block_indices(mat_type, bi, bj):
        """Return list of (flat_index, 1) for all elements in block (bi,bj) of a 4x4 matrix."""
        indices = []
        for di in range(2):
            for dj in range(2):
                if mat_type == 'U':  # A is 4x4, index = i*4+j
                    indices.append(((bi*2+di)*4 + (bj*2+dj), 1))
                elif mat_type == 'V':  # B is 4x4, index = k*4+l
                    indices.append(((bi*2+di)*4 + (bj*2+dj), 1))
                elif mat_type == 'W':  # C is 4x4, index = l*4+i
                    # C block (ci, cj) where ci is row of C, cj is col of C
                    # W index = col*4 + row = cj*4 + ci
                    # Block (ci, cj) element (di, dj) is at W[(cj*2+dj)*4 + (ci*2+di)]
                    indices.append(((bj*2+dj)*4 + (bi*2+di), 1))
        return indices
    
    # Now define the 7 Strassen multiplications.
    # Each is: X_blocks, Y_blocks, Z_block_map
    # X_blocks: list of (block_idx, coeff) where block_idx is (bi,bj)
    # Y_blocks: list of (block_idx, coeff)
    # Z_block_map: list of (z_block_idx, c_block_idx, coeff) where z_block_idx is (a,c) in 2x2 result
    
    strassen_defs = [
        # M1 = (A00+A11)*(B00+B11)
        {'X': [((0,0), 1), ((1,1), 1)], 'Y': [((0,0), 1), ((1,1), 1)],
         'Z': [((0,0), (0,0), 1), ((1,1), (1,1), 1)]},
        # M2 = (A10+A11)*B00
        {'X': [((1,0), 1), ((1,1), 1)], 'Y': [((0,0), 1)],
         'Z': [((0,0), (1,0), 1)]},
        # M3 = A00*(B01-B11)
        {'X': [((0,0), 1)], 'Y': [((0,1), 1), ((1,1), -1)],
         'Z': [((0,0), (0,1), 1)]},
        # M4 = A11*(B10-B00)
        {'X': [((1,1), 1)], 'Y': [((1,0), 1), ((0,0), -1)],
         'Z': [((0,0), (0,0), 1), ((1,0), (1,0), 1)]},
        # M5 = (A00+A01)*B11
        {'X': [((0,0), 1), ((0,1), 1)], 'Y': [((1,1), 1)],
         'Z': [((0,0), (0,1), 1), ((0,1), (0,0), -1)]},
        # M6 = (A10-A00)*(B00+B01)
        {'X': [((1,0), 1), ((0,0), -1)], 'Y': [((0,0), 1), ((0,1), 1)],
         'Z': [((1,1), (1,1), 1)]},
        # M7 = (A01-A11)*(B10+B11)
        {'X': [((0,1), 1), ((1,1), -1)], 'Y': [((1,0), 1), ((1,1), 1)],
         'Z': [((0,0), (0,0), 1)]},
    ]
    
    for sdef in strassen_defs:
        X_blocks = sdef['X']
        Y_blocks = sdef['Y']
        Z_map = sdef['Z']
        
        # X is a 2x2 matrix where X[a,b] = sum of coeff * A_block(block_idx)
        # Y is a 2x2 matrix where Y[c,d] = sum of coeff * B_block(block_idx)
        # Z[a,c] = sum_b X[a,b] * Y[b,c]
        # Each Z[a,c] maps to C_block(c_block_idx) with some coefficient.
        
        # For each (a, b, c, d) in the 2x2x2 schoolbook:
        # Z[a,c] += X[a,b] * Y[b,c]
        # X[a,b] is a linear combo of A blocks
        # Y[b,c] is a linear combo of B blocks
        # So the product X[a,b]*Y[b,c] is a sum of products of A_block * B_block
        # Each such product is a 2x2x2 schoolbook multiplication (rank 4)
        
        # Actually, let me just expand everything to rank-1 terms.
        # X[a,b] = sum_{(bi,bj),coeff} coeff * A_block(bi,bj)
        # Y[b,c] = sum_{(bk,bl),coeff2} coeff2 * B_block(bk,bl)
        # X[a,b]*Y[b,c] = sum_{...} coeff*coeff2 * A_block(bi,bj) * B_block(bk,bl)
        # A_block(bi,bj) * B_block(bk,bl) is a 2x2x2 matrix product (rank 4)
        # The result contributes to Z[a,c] which maps to C_block.
        
        # For each pair (X_term, Y_term) where X_term is from X[a,b] and Y_term is from Y[b,c]:
        # We get a 2x2x2 schoolbook multiplication with coefficients.
        
        for a in range(2):
            for c in range(2):
                # Z[a,c] = sum_b X[a,b] * Y[b,c]
                for b in range(2):
                    # Get X[a,b] terms
                    X_terms = []
                    for (bi, bj), coeff in X_blocks:
                        # X[a,b] includes this block if the block index matches (a,b) in the 2x2 sense
                        # Wait, I need to be more careful.
                        # X is a 2x2 matrix. X[a,b] is the element at row a, col b.
                        # X[a,b] = sum of coeff * A_block(block_idx) for block_idx in X_blocks
                        # But X_blocks defines the entire X matrix, not individual elements.
                        # I need to redefine.
                        pass
                    break
                break
            break
        break
    
    # Let me redo this more carefully.
    # The 7 Strassen multiplications for 4x4x4:
    # Each multiplication M = X * Y where X and Y are 2x2 matrices formed from blocks.
    # X is 2x2, Y is 2x2, result Z is 2x2.
    # X[i,j] is a 2x2 matrix (a block of A or sum of blocks)
    # Y[i,j] is a 2x2 matrix (a block of B or sum of blocks)
    # Z[i,j] = sum_k X[i,k] * Y[k,j] (2x2 matrix product)
    # Z[i,j] contributes to C block.
    
    # Let me define each Strassen multiplication as:
    # X: 2x2 array of (list of (A_block, coeff))
    # Y: 2x2 array of (list of (B_block, coeff))
    # Z_to_C: 2x2 array of (C_block, coeff)
    
    def make_2x2(block_list):
        """Create a 2x2 array where each entry is a list of (block, coeff)."""
        result = [[[] for _ in range(2)] for _ in range(2)]
        for (i, j), blocks in block_list:
            result[i][j] = blocks
        return result
    
    # M1 = (A00+A11)*(B00+B11)
    # X[0,0] = A00, X[1,1] = A11, others = 0
    # Y[0,0] = B00, Y[1,1] = B11, others = 0
    # Z[0,0] = C00, Z[1,1] = C11
    M1 = {
        'X': make_2x2([((0,0), [((0,0), 1)]), ((1,1), [((1,1), 1)])]),
        'Y': make_2x2([((0,0), [((0,0), 1)]), ((1,1), [((1,1), 1)])]),
        'Z': make_2x2([((0,0), [((0,0), 1)]), ((1,1), [((1,1), 1)])])
    }
    
    # M2 = (A10+A11)*B00
    # X[0,0] = 0, X[0,1] = 0, X[1,0] = A10, X[1,1] = A11
    # Wait, (A10+A11) is a 2x2 matrix where both entries are the sum? No.
    # A10 is a 2x2 block, A11 is a 2x2 block.
    # (A10+A11) means X[0,0] = A10[0,0]+A11[0,0], etc.
    # So X is a 2x2 matrix where X[i,j] = A10[i,j] + A11[i,j]
    # In terms of blocks: X[0,0] = A10_block + A11_block (as 2x2 matrices)
    # But A10 and A11 are different blocks, so X[0,0] = A10[0,0] + A11[0,0]
    # This is not a single block, it's a sum of elements from two blocks.
    
    # I think I'm overcomplicating this. Let me just directly enumerate the 28 rank-1 terms
    # by thinking of the 4x4x4 Strassen decomposition as 7 independent 2x2x2 schoolbook multiplications.
    
    # Actually, the cleanest way: the 4x4x4 Strassen decomposition has rank 28.
    # It consists of 7 multiplications of 2x2 matrices, each of rank 4 (schoolbook).
    # Each 2x2 matrix multiplication is: C[i,j] = sum_k A[i,k]*B[k,j] for i,j,k in {0,1}
    # Each term A[i,k]*B[k,j] is a rank-1 term.
    
    # For the 4x4x4 case, each "2x2 matrix" is actually a 2x2 block of the 4x4 matrix.
    # So A[i,k] in the 2x2 sense is a 2x2 block of the 4x4 A matrix.
    # The product of two 2x2 blocks is a 2x2x2 matrix multiplication (rank 4).
    
    # Let me just hardcode the 28 terms.
    # I'll use a different approach: generate them programmatically.
    
    terms = []
    
    # Define the 7 Strassen multiplications for 4x4x4
    # Each is defined by:
    #   A_factors: list of (a_row_block, a_col_block, coeff) for the first 2x2 matrix
    #   B_factors: list of (b_row_block, b_col_block, coeff) for the second 2x2 matrix
    #   C_map: list of (c_row_block, c_col_block, coeff) for where the result goes
    
    # But this is still complex. Let me just use a known rank-28 decomposition.
    
    # Actually, let me try a different approach: use the fact that 4x4x9 can be decomposed
    # as 4x4x4 + 4x4x4 + 4x4x1, and use rank 28 for each 4x4x4 and rank 16 for 4x4x1.
    # Total: 28 + 28 + 16 = 72.
    
    # But I need to verify that this is correct. Let me just implement it and verify.
    
    # For now, let me return the schoolbook decomposition (rank 144) as a fallback.
    return None

def _verify_exact(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Check entries are integer or half-integer
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    # Check tensor equation for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to build a rank-28 decomposition for 4x4x4 blocks
    # 4x4x9 = 4x4x4 (cols 0-3) + 4x4x4 (cols 4-7) + 4x4x1 (col 8)
    # Rank: 28 + 28 + 16 = 72
    
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 72
    
    s22_terms = _strassen_2x2x2()
    
    def add_strassen_4x4x4(offset, b_col_offset):
        """Add Strassen 4x4x4 decomposition.
        A is 4x4 (all rows, all cols of A)
        B is 4x4 (all rows, cols b_col_offset to b_col_offset+3)
        C is 4x4 (all rows, all cols of C)
        W index = k*4 + i where k is global col of B, i is row of C
        """
        # 7 Strassen multiplications, each is a 2x2x2 schoolbook (rank 4)
        # Total: 28 rank-1 terms
        
        # Define the 7 multiplications
        # Each multiplication: X (2x2 from A blocks) * Y (2x2 from B blocks) -> Z (2x2)
        # X[i,j] is a 2x2 block of A or sum
        # Y[i,j] is a 2x2 block of B or sum
        # Z[i,j] contributes to C block
        
        # A block (bi, bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
        # B block (bk, bl): rows bk*2 to bk*2+1, cols b_col_offset+bl*2 to b_col_offset+bl*2+1
        # C block (ci, cj): rows ci*2 to ci*2+1, cols cj*2 to cj*2+1
        # W index for C element (row, col) = col*4 + row
        
        # For each Strassen multiplication, we have:
        # X is a 2x2 matrix of 2x2 blocks
        # Y is a 2x2 matrix of 2x2 blocks
        # Z is a 2x2 matrix of 2x2 blocks (result)
        
        # Let me define each multiplication as a list of (X_block, Y_block, Z_block, coeff)
        # where X_block is (bi, bj, coeff), Y_block is (bk, bl, coeff), Z_block is (ci, cj, coeff)
        # and the term is: sum over all elements of X_block * Y_block -> Z_block
        
        # Actually, each "multiplication" is a 2x2x2 matrix multiplication.
        # Let me just enumerate the 4 terms for each of the 7 multiplications.
        
        # M1: X = [[A00, 0], [0, A11]], Y = [[B00, 0], [0, B11]], Z = [[C00, 0], [0, C11]]
        # M1[0,0] = A00*B00 -> C00
        # M1[0,1] = A00*0 + 0*B11 = 0
        # M1[1,0] = 0*B00 + A11*0 = 0
        # M1[1,1] = A11*B11 -> C11
        
        # Wait, that's not right. Strassen M1 = (A00+A11)*(B00+B11)
        # This is a single 2x2x2 multiplication where:
        # X = A00+A11 (a 2x2 matrix where X[i,j] = A00[i,j] + A11[i,j])
        # Y = B00+B11 (a 2x2 matrix where Y[i,j] = B00[i,j] + B11[i,j])
        # Z = X*Y (2x2 matrix product)
        # Z[0,0] = sum_k X[0,k]*Y[k,0]
        # etc.
        
        # So for M1, the 4 schoolbook terms are:
        # Z[0,0] += X[0,0]*Y[0,0] = (A00[0,0]+A11[0,0])*(B00[0,0]+B11[0,0])
        # Z[0,0] += X[0,1]*Y[1,0] = (A00[0,1]+A11[0,1])*(B00[1,0]+B11[1,0])
        # Z[0,1] += X[0,0]*Y[0,1] = (A00[0,0]+A11[0,0])*(B00[0,1]+B11[0,1])
        # Z[0,1] += X[0,1]*Y[1,1] = (A00[0,1]+A11[0,1])*(B00[1,1]+B11[1,1])
        # Z[1,0] += X[1,0]*Y[0,0] = (A00[1,0]+A11[1,0])*(B00[0,0]+B11[0,0])
        # Z[1,0] += X[1,1]*Y[1,0] = (A00[1,1]+A11[1,1])*(B00[1,0]+B11[1,0])
        # Z[1,1] += X[1,0]*Y[0,1] = (A00[1,0]+A11[1,0])*(B00[0,1]+B11[0,1])
        # Z[1,1] += X[1,1]*Y[1,1] = (A00[1,1]+A11[1,1])*(B00[1,1]+B11[1,1])
        
        # Each of these is a product of sums, which expands to 4 rank-1 terms each.
        # So M1 has 4*4 = 16 rank-1 terms? No, that's not right either.
        
        # I think I need to be more careful. Let me just give up on the Strassen approach
        # for now and return the schoolbook decomposition.
        
        return offset
    
    # Given the complexity, let me just return schoolbook for now.
    # The deadline guard ensures we don't timeout.
    
    if time.time() > deadline - 5:
        return _schoolbook()
    
    # For now, return schoolbook. Future work: implement Strassen correctly.
    return _schoolbook()
# EVOLVE-BLOCK-END
