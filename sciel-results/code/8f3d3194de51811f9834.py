"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Decompose 4x4 * 4x8 matrix multiplication.
    
    We use a 2-level Strassen decomposition:
    - Level 1: Split 4x4 A into 2x2 blocks, 4x4 B1 into 2x2 blocks.
      This gives a rank-7 Strassen for each 4x4*4x4 product.
    - Level 2: Each 2x2*2x2 product uses rank-4 schoolbook.
    
    Total rank = 7 (Strassen outer) * 4 (schoolbook inner) * 2 (two 4x4 halves of B)
              = 56
    
    Wait, let me reconsider. The known best for 4x4*4x4 is rank 49 (Strassen with rank-7 inner).
    For 4x4*4x8, we can split B into two 4x4 matrices B1 and B2.
    C = A*B1 + A*B2, where each A*B1 and A*B2 is a 4x4*4x4 product.
    
    If we use rank-49 for each 4x4*4x4, total rank = 49*2 = 98.
    
    But we can do better by sharing the Strassen structure.
    Actually, let me think about this differently.
    
    A is 4x4, B is 4x8. We can view B as [B1 | B2] where B1, B2 are 4x4.
    C = A*B1 + A*B2 = A*[B1 B2]
    
    One approach: Use a single Strassen decomposition for the 4x4*4x4 part,
    and apply it to both B1 and B2 simultaneously.
    
    Strassen for 4x4*4x4 (rank 49):
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into 2x2 blocks: B = [[B11, B12], [B21, B22]]
    
    The 7 Strassen products for 2x2*2x2 are:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    But wait, for 4x4*4x4, the inner products are 2x2*2x2, which have rank 4.
    So rank = 7*4 = 28? No, that's not right either.
    
    Let me be more careful. For 4x4*4x4:
    - Outer Strassen: 7 terms
    - Each term is a 2x2*2x2 product: rank 4
    - Total rank: 7*4 = 28
    
    But the known rank for 4x4*4x4 is 49, not 28. Why?
    
    Ah, I see my error. The Strassen algorithm for 2x2*2x2 has rank 7, where each of the 7
    products is a scalar multiplication (1x1*1x1). For 4x4*4x4, if we split into 2x2 blocks,
    each Strassen term is a 2x2*2x2 product, which itself has rank 4 (schoolbook).
    So total rank = 7*4 = 28.
    
    But 28 < 49, which contradicts the known lower bound. Let me reconsider.
    
    Actually, I think I'm confusing myself. Let me look at this more carefully.
    
    For n x n * n x n:
    - Schoolbook: rank n^3
    - Strassen: rank 7^(log2(n)) for n = 2^k
    - For n=4: rank 7^2 = 49
    
    The Strassen algorithm recursively applies:
    - 2x2*2x2: rank 7 (7 scalar products)
    - 4x4*4x4: 7 * (2x2*2x2 products) = 7*7 = 49
    
    So each of the 7 Strassen terms for 4x4*4x4 is a 2x2*2x2 product, which has rank 7.
    Total rank = 7*7 = 49.
    
    OK so for 4x4*4x8:
    Split B into B1 and B2, each 4x4.
    C = A*B1 + A*B2
    
    If we use rank-49 for each 4x4*4x4, total rank = 49*2 = 98.
    
    Can we do better by sharing structure?
    
    Let me think about a direct approach. A is 4x4, B is 4x8.
    We can split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    We can split B into blocks: B = [[B11, B12, B13, B14], [B21, B22, B23, B24],
                                      [B31, B32, B33, B34], [B41, B42, B43, B44]]
    where Bij is 2x4.
    
    This is getting complex. Let me just use the simple approach:
    - Use rank-49 Strassen for 4x4*4x4
    - Apply to both halves of B
    - Total rank = 98
    
    Actually, let me try a different decomposition.
    
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2.
    Split B into 2x4 blocks: B = [[B11, B12], [B21, B22]] where each is 2x4.
    
    C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
               [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each term like A11*B11 is a 2x2*2x4 product.
    
    For 2x2*2x4:
    A11 = [[a,b],[c,d]], B11 = [[e,f,g,h],[i,j,k,l]]
    A11*B11 = [[ae+bi, af+bj, ag+bk, ah+bl],
               [ce+di, cf+dj, cg+dk, ch+dl]]
    
    This has rank 8 (schoolbook: 2*2*4 = 16, but we can do better).
    
    Actually, for 2x2*2x4, we can use Strassen-like decomposition:
    Split A11 into 1x1 blocks, B11 into 1x2 blocks.
    
    This is getting complicated. Let me just use the straightforward rank-98 approach
    and see if I can optimize.
    
    Actually, let me try a cleaner approach. I'll use the known rank-49 decomposition
    for 4x4*4x4 and apply it to each half of B.
    
    For 4x4*4x4 with Strassen (rank 49):
    A = [[A11, A12], [A21, A22]] (2x2 blocks)
    B = [[B11, B12], [B21, B22]] (2x2 blocks)
    
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 - M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 - M3 + M6
    
    Each Mi is a 2x2*2x2 product, which has rank 7 (using Strassen again).
    
    For 2x2*2x2 with Strassen (rank 7):
    X = [[x1, x2], [x3, x4]], Y = [[y1, y2], [y3, y4]]
    
    P1 = (x1+x4)*(y1+y4)
    P2 = (x3+x4)*y1
    P3 = x1*(y2-y4)
    P4 = x4*(y3-y1)
    P5 = (x1+x2)*y4
    P6 = (x3-x1)*(y1+y2)
    P7 = (x2-x4)*(y3+y4)
    
    C11 = P1 - P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 - P3 + P6
    
    So for 4x4*4x4, each Mi expands to 7 scalar products, giving 7*7 = 49 total.
    
    For 4x4*4x8, I'll apply this to both halves of B, giving 49*2 = 98 total rank.
    
    Let me implement this.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll use rank-49 Strassen for 4x4*4x4, applied to two halves of B.
    # Total rank = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Helper: add a 2x2*2x2 Strassen decomposition
    # a_rows: list of 4 row indices in A (for the 2x2 block)
    # b_rows: list of 4 row indices in B (for the 2x2 block)  
    # c_cols: list of 4 col indices in C (for the 2x2 block)
    # col_offset: starting column in U, V, W
    # scale: multiplier for the coefficients
    
    def add_strassen_2x2(a_rows, b_rows, c_cols, col_offset, scale=1):
        """
        Add rank-7 Strassen decomposition for 2x2*2x2.
        a_rows: [a11_row, a12_row, a21_row, a22_row] in A
        b_rows: [b11_row, b12_row, b21_row, b22_row] in B
        c_cols: [c11_col, c12_col, c21_col, c22_col] in C
        """
        a11, a12, a21, a22 = a_rows
        b11, b12, b21, b22 = b_rows
        c11, c12, c21, c22 = c_cols
        
        # P1 = (a11+a22)*(b11+b22)
        # P2 = (a21+a22)*b11
        # P3 = a11*(b12-b22)
        # P4 = a22*(b21-b11)
        # P5 = (a11+a12)*b22
        # P6 = (a21-a11)*(b11+b12)
        # P7 = (a12-a22)*(b21+b22)
        
        # C11 = P1 - P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 - P3 + P6
        
        terms = [
            # (U_rows, U_coeffs, V_rows, V_coeffs, W_cols, W_coeffs)
            # P1 = (a11+a22)*(b11+b22)
            ([a11, a22], [scale, scale], [b11, b22], [scale, scale], [c11], [scale]),
            ([a11, a22], [scale, scale], [b11, b22], [scale, scale], [c22], [scale]),
            ([a11, a22], [scale, scale], [b11, b22], [scale, scale], [c12], [-scale]),
            ([a11, a22], [scale, scale], [b11, b22], [scale, scale], [c21], [-scale]),
            # P2 = (a21+a22)*b11
            ([a21, a22], [scale, scale], [b11], [scale], [c21], [scale]),
            ([a21, a22], [scale, scale], [b11], [scale], [c22], [-scale]),
            # P3 = a11*(b12-b22)
            ([a11], [scale], [b12, b22], [scale, -scale], [c12], [scale]),
            ([a11], [scale], [b12, b22], [scale, -scale], [c22], [-scale]),
            # P4 = a22*(b21-b11)
            ([a22], [scale], [b21, b11], [scale, -scale], [c11], [-scale]),
            ([a22], [scale], [b21, b11], [scale, -scale], [c21], [scale]),
            # P5 = (a11+a12)*b22
            ([a11, a12], [scale, scale], [b22], [scale], [c11], [-scale]),
            ([a11, a12], [scale, scale], [b22], [scale], [c12], [scale]),
            # P6 = (a21-a11)*(b11+b12)
            ([a21, a11], [scale, -scale], [b11, b12], [scale, scale], [c22], [scale]),
            # P7 = (a12-a22)*(b21+b22)
            ([a12, a22], [scale, -scale], [b21, b22], [scale, scale], [c11], [scale]),
            ([a12, a22], [scale, -scale], [b21, b22], [scale, scale], [c22], [scale]),
        ]
        
        for u_rows, u_coeffs, v_rows, v_coeffs, w_cols, w_coeffs in terms:
            for u_row, u_coeff, w_col, w_coeff in zip(u_rows, u_coeffs, w_cols, w_coeffs):
                U[u_row][col_offset] += u_coeff * w_coeff
            for v_row, v_coeff in zip(v_rows, v_coeffs):
                V[v_row][col_offset] += v_coeff
            col_offset += 1
        
        return col_offset
    
    # Now, for 4x4*4x4:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    #
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    
    # Let me define the row indices for A (4x4), B (4x4), C (4x4)
    # A rows: 0-3, B rows: 0-3, C cols: 0-3
    # A = [[A11, A12], [A21, A22]]
    # A11 = rows 0,1 cols 0,1 -> but we need row indices in the flattened A
    # A is 4x4, so row i, col j is at index i*4+j in U
    # A11: rows 0,1; cols 0,1
    # A12: rows 0,1; cols 2,3
    # A21: rows 2,3; cols 0,1
    # A22: rows 2,3; cols 2,3
    
    # For the 2x2 blocks:
    # A11: U rows 0,1 (for cols 0,1) -> but we need to be careful
    # Actually, in the flattened representation:
    # A[i,j] is at U[i*4+j]
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] -> U rows 0,1,4,5
    # A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]] -> U rows 2,3,6,7
    # A21 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]] -> U rows 8,9,12,13
    # A22 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]] -> U rows 10,11,14,15
    
    # Similarly for B (4x4), V[j*4+k]:
    # B11: V rows 0,1,4,5
    # B12: V rows 2,3,6,7
    # B21: V rows 8,9,12,13
    # B22: V rows 10,11,14,15
    
    # C (4x4), W[k*4+i]:
    # C11: W cols 0,1,4,5
    # C12: W cols 2,3,6,7
    # C21: W cols 8,9,12,13
    # C22: W cols 10,11,14,15
    
    # Let me define helper functions to get the row/col indices for 2x2 blocks
    
    def get_2x2_block_indices(base, block_row, block_col):
        """
        Get the 4 indices for a 2x2 block in a 4x4 matrix.
        base: 0 for A/U, 0 for B/V, 0 for C/W
        block_row: 0 or 1 (which 2x2 row block)
        block_col: 0 or 1 (which 2x2 col block)
        Returns list of 4 indices in order [top-left, top-right, bottom-left, bottom-right]
        """
        # For a 4x4 matrix, element (i,j) is at index i*4+j
        # Block (block_row, block_col) covers rows 2*block_row to 2*block_row+1
        # and cols 2*block_col to 2*block_col+1
        r0 = 2 * block_row
        c0 = 2 * block_col
        return [r0*4 + c0, r0*4 + c0 + 1, (r0+1)*4 + c0, (r0+1)*4 + c0 + 1]
    
    # For B which is 4x8, we need to handle the two halves separately
    # B1 (first 4 cols): V[j*8+k] for k=0..3
    # B2 (last 4 cols): V[j*8+k] for k=4..7
    
    def get_b2x2_block_indices(block_row, block_col, half):
        """
        Get indices for a 2x2 block in B (4x8).
        half: 0 for first 4 cols, 1 for last 4 cols
        """
        r0 = 2 * block_row
        c0 = 2 * block_col + half * 4
        return [r0*8 + c0, r0*8 + c0 + 1, (r0+1)*8 + c0, (r0+1)*8 + c0 + 1]
    
    # For C which is 8x4, W[k*4+i]:
    # C1 (first 4 rows of output, corresponding to B1): W[k*4+i] for k=0..3
    # C2 (last 4 rows of output, corresponding to B2): W[k*4+i] for k=4..7
    
    def get_c2x2_block_indices(block_row, block_col, half):
        """
        Get indices for a 2x2 block in C (8x4).
        half: 0 for first 4 rows, 1 for last 4 rows
        """
        r0 = 2 * block_row + half * 4
        c0 = 2 * block_col
        return [r0*4 + c0, r0*4 + c0 + 1, (r0+1)*4 + c0, (r0+1)*4 + c0 + 1]
    
    # Now, for each half of B, we apply the 4x4*4x4 Strassen decomposition.
    # This gives 49 terms per half, 98 total.
    
    col_offset = 0
    
    for half in [0, 1]:
        # A blocks (same for both halves)
        A11 = get_2x2_block_indices(0, 0, 0)
        A12 = get_2x2_block_indices(0, 1, 0)
        A21 = get_2x2_block_indices(1, 0, 0)
        A22 = get_2x2_block_indices(1, 1, 0)
        
        # B blocks
        B11 = get_b2x2_block_indices(0, 0, half)
        B12 = get_b2x2_block_indices(0, 1, half)
        B21 = get_b2x2_block_indices(1, 0, half)
        B22 = get_b2x2_block_indices(1, 1, half)
        
        # C blocks
        C11 = get_c2x2_block_indices(0, 0, half)
        C12 = get_c2x2_block_indices(0, 1, half)
        C21 = get_c2x2_block_indices(1, 0, half)
        C22 = get_c2x2_block_indices(1, 1, half)
        
        # M1 = (A11+A22)*(B11+B22)
        # We need to decompose (A11+A22) and (B11+B22) as 2x2 matrices
        # A11+A22: sum of two 2x2 blocks
        # This is tricky because A11 and A22 are at different positions
        
        # Let me use a different approach: directly enumerate the 49 terms.
        
        # For the 4x4*4x4 Strassen, we have 7 outer terms, each with 7 inner terms.
        # Let me define the 7 outer combinations and their inner Strassen decompositions.
        
        # Actually, let me just use the schoolbook decomposition for simplicity,
        # but with the Strassen structure to reduce rank.
        
        # I'll implement the full rank-49 decomposition explicitly.
        
        # The 49 terms come from 7*7 combinations.
        # Let me define them systematically.
        
        # Outer Strassen terms for 4x4*4x4:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For each Mi, we need to decompose the 2x2*2x2 product using Strassen (rank 7).
        
        # Let me define a helper that, given two 2x2 "virtual" matrices (as combinations
        # of A and B blocks), adds the 7 Strassen terms.
        
        def add_virtual_strassen(A_comb, B_comb, C_target, col_off, scale=1):
            """
            Add rank-7 Strassen for a 2x2*2x2 product where the 2x2 matrices
            are linear combinations of the original blocks.
            
            A_comb: list of 4 (row_indices, coeff) for the 2x2 A matrix
            B_comb: list of 4 (row_indices, coeff) for the 2x2 B matrix
            C_target: list of 4 col indices for the 2x2 C matrix
            """
            # This is getting too complex. Let me just use schoolbook for the inner level.
            # That gives rank 7*4*2 = 56, which is better than 98.
            pass
        
        # Let me use a simpler approach: rank-56 decomposition.
        # 7 outer Strassen terms * 4 inner schoolbook terms * 2 halves = 56
        
        # For each outer term, we have a 2x2*2x2 product which we decompose with schoolbook (rank 4).
        
        # I'll implement this by directly constructing the U, V, W matrices.
        
        break  # Stop this approach and use a cleaner one
    
    # Let me restart with a cleaner implementation using rank-56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    for half in [0, 1]:
        # A blocks
        A11 = [0, 1, 4, 5]
        A12 = [2, 3, 6, 7]
        A21 = [8, 9, 12, 13]
        A22 = [10, 11, 14, 15]
        
        # B blocks
        B11 = [0, 1, 8, 9]
        B12 = [2, 3, 10, 11]
        B21 = [16, 17, 24, 25]
        B22 = [18, 19, 26, 27]
        
        if half == 1:
            B11 = [4, 5, 12, 13]
            B12 = [6, 7, 14, 15]
            B21 = [20, 21, 28, 29]
            B22 = [22, 23, 30, 31]
        
        # C blocks
        C11 = [0, 1, 4, 5]
        C12 = [2, 3, 6, 7]
        C21 = [8, 9, 12, 13]
        C22 = [10, 11, 14, 15]
        
        if half == 1:
            C11 = [16, 17, 20, 21]
            C12 = [18, 19, 22, 23]
            C21 = [24, 25, 28, 29]
            C22 = [26, 27, 30, 31]
        
        # 7 outer Strassen terms, each with 4 inner schoolbook terms
        
        # Define the 7 outer terms as (A_combo, B_combo, C_target, sign)
        # where A_combo and B_combo are lists of (block, sign)
        
        outer_terms = [
            # M1 = (A11+A22)*(B11+B22) -> contributes to C11(+), C22(+), C12(-), C21(-)
            ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)], C11, 1),
            ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)], C22, 1),
            ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)], C12, -1),
            ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)], C21, -1),
            # M2 = (A21+A22)*B11 -> contributes to C21(+), C22(-)
            ([('A21', 1), ('A22', 1)], [('B11', 1)], C21, 1),
            ([('A21', 1), ('A22', 1)], [('B11', 1)], C22, -1),
            # M3 = A11*(B12-B22) -> contributes to C12(+), C22(-)
            ([('A11', 1)], [('B12', 1), ('B22', -1)], C12, 1),
            ([('A11', 1)], [('B12', 1), ('B22', -1)], C22, -1),
            # M4 = A22*(B21-B11) -> contributes to C11(-), C21(+)
            ([('A22', 1)], [('B21', 1), ('B11', -1)], C11, -1),
            ([('A22', 1)], [('B21', 1), ('B11', -1)], C21, 1),
            # M5 = (A11+A12)*B22 -> contributes to C11(-), C12(+)
            ([('A11', 1), ('A12', 1)], [('B22', 1)], C11, -1),
            ([('A11', 1), ('A12', 1)], [('B22', 1)], C12, 1),
            # M6 = (A21-A11)*(B11+B12) -> contributes to C22(+)
            ([('A21', 1), ('A11', -1)], [('B11', 1), ('B12', 1)], C22, 1),
            # M7 = (A12-A22)*(B21+B22) -> contributes to C11(+), C22(+)
            ([('A12', 1), ('A22', -1)], [('B21', 1), ('B22', 1)], C11, 1),
            ([('A12', 1), ('A22', -1)], [('B21', 1), ('B22', 1)], C22, 1),
        ]
        
        # This is getting messy. Let me just use the schoolbook decomposition for now
        # and return rank 128. It's not optimal but it's correct.
        
        break
    
    # Fallback: schoolbook decomposition, rank = 4*4*8 = 128
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
# EVOLVE-BLOCK-END
