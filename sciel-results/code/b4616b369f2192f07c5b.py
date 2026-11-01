"""Program implementing Strassen-based tensor decomposition for 2x3 * 3x7."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2(A, B):
    """
    Returns list of 7 (u, v, w) tuples for the 7 multiplication terms of Strassen's algorithm.
    A, B are 2x2 lists of symbolic variables (strings).
    Each u and v is a list of (coeff, var) pairs.
    w is a list of (coeff, term_index, target) where target is (i,j) in C.
    """
    # Define addition of symbolic expressions
    def add_expr(e1, e2):
        from collections import defaultdict
        d = defaultdict(int)
        for c, v in e1:
            d[v] += c
        for c, v in e2:
            d[v] += c
        return [(c, v) for v, c in d.items() if c != 0]

    def sub_expr(e1, e2):
        from collections import defaultdict
        d = defaultdict(int)
        for c, v in e1:
            d[v] += c
        for c, v in e2:
            d[v] -= c
        return [(c, v) for v, c in d.items() if c != 0]

    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    a11 = [(1, A[0][0])]
    a12 = [(1, A[0][1])]
    a21 = [(1, A[1][0])]
    a22 = [(1, A[1][1])]
    b11 = [(1, B[0][0])]
    b12 = [(1, B[0][1])]
    b21 = [(1, B[1][0])]
    b22 = [(1, B[1][1])]

    # Strassen terms
    M1 = add_expr(a11, a22)  # A11 + A22
    M2 = add_expr(a21, a22)  # A21 + A22
    M3 = add_expr(a11, a12)  # A11 + A12
    M4 = add_expr(a21, a22)  # A21 + A22 (same as M2? No, standard is M4 = A21 - A11? Let's use standard)
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    u1 = add_expr(a11, a22)
    v1 = add_expr(b11, b22)
    w1 = [(1, 0, (0,0)), (1, 0, (1,1))] # M1 contributes to C11 (+) and C22 (+)

    u2 = add_expr(a21, a22)
    v2 = b11
    w2 = [(1, 1, (1,0)), (-1, 1, (1,1))] # M2 contributes to C21 (+) and C22 (-)

    u3 = a11
    v3 = sub_expr(b12, b22)
    w3 = [(1, 2, (0,1)), (1, 2, (1,1))] # M3 contributes to C12 (+) and C22 (+)

    u4 = a22
    v4 = sub_expr(b21, b11)
    w4 = [(1, 3, (0,0)), (1, 3, (1,0))] # M4 contributes to C11 (+) and C21 (+)

    u5 = add_expr(a11, a12)
    v5 = b22
    w5 = [(-1, 4, (0,0)), (1, 4, (0,1))] # M5 contributes to C11 (-) and C12 (+)

    u6 = sub_expr(a21, a11)
    v6 = add_expr(b11, b12)
    w6 = [(1, 5, (1,1))] # M6 contributes to C22 (+)

    u7 = sub_expr(a12, a22)
    v7 = add_expr(b21, b22)
    w7 = [(1, 6, (0,0))] # M7 contributes to C11 (+)

    return [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7),
    ]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Split B (3x7) into two blocks: B_left (3x4) and B_right (3x3)
    # B_left is split into two 3x2 blocks: BL1, BL2
    # B_right is split into one 3x2 block BR1 and one 3x1 block BR2? 
    # Wait, the directive says "grouping the seven columns of the B-matrix into two blocks (sizes 4 and 3)".
    # And "applying Strassen's 2x2 block algorithm to each sub-block independently".
    # Strassen works on 2x2 blocks. 
    # If we have A (2x3) and B (3x4), we can split A into A_left (2x2) and A_right (2x1).
    # B_left (3x4) into B1 (3x2) and B2 (3x2).
    # This doesn't fit Strassen directly on 2x2 blocks unless we pad.
    
    # Alternative interpretation:
    # The "Four Russians" bit-packing hint suggests grouping.
    # Let's look at the dimensions: A is 2x3, B is 3x7.
    # If we view A as [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    # And B as [B1 B2 B3 B4 B5 B6 B7] where each Bi is 3x1.
    
    # The directive specifically says: "grouping the seven columns of the B-matrix into two blocks (sizes 4 and 3)".
    # Let B1 be the first 4 columns (3x4) and B2 be the last 3 columns (3x3).
    # For B1 (3x4): We can split A into A1 (2x2) and A2 (2x1).
    #   C1 = A1 * B1_left + A2 * B1_right
    #   Where B1_left is 3x2 and B1_right is 3x2.
    #   This is a 2x2 * 3x2 plus 2x1 * 3x2.
    #   This doesn't immediately look like a 2x2 Strassen.
    
    # Let's reconsider the "Strassen's 2x2 block algorithm" on sub-blocks.
    # Perhaps the matrix A (2x3) is padded to 2x4? Or B (3x7) is padded?
    # If we pad A to 2x4 (adding a zero column), and B to 3x8 (adding one zero column).
    # Then A is [A11 A12 | A21 A22] where each is 2x2.
    # B is [B11 B12; B21 B22] where each is 2x2? No, B is 3x8.
    # Strassen requires square blocks.
    
    # Let's try a different grouping.
    # What if we group the 3 rows of A and B differently?
    # No, A is 2x3, B is 3x7.
    
    # Let's look at the rank claim: 18 = 2*2^3 + 2*2^2 = 16 + 8? No, 2*8 + 2*4 = 16+8=24? No.
    # 2*2^3 = 16. 2*2^2 = 8. Total 24. The directive says 18.
    # Maybe 2 blocks of size 4x4? Rank 2*7 = 14?
    # Maybe 1 block of size 2x2 (rank 7) and others?
    
    # Let's ignore the specific rank number in the directive for a moment and focus on the structure.
    # "Four Russians" bit-packing usually refers to precomputing results for small chunks.
    # Here, "bit-packing" might mean representing the multiplication as a combination of precomputed products.
    
    # Let's try to implement a valid decomposition first, then optimize.
    # Schoolbook rank is 42.
    # We can split B into two parts: B1 (3x4) and B2 (3x3).
    # C1 = A * B1 (2x4)
    # C2 = A * B2 (2x3)
    #
    # For C1 = A * B1:
    # A is 2x3. B1 is 3x4.
    # We can split A into A1 (2x2) and A2 (2x1).
    # B1 into B1a (3x2) and B1b (3x2).
    # C1 = [A1 A2] * [B1a B1b] = [A1*B1a + A2*B1a  A1*B1b + A2*B1b]? No.
    # C1 = A1 * B1a + A2 * B1a? No.
    # C1 = A1 * [B1a B1b] + A2 * [B1a B1b] ? No.
    # C1 = A1 * B1_left_half + A2 * B1_right_half? No, A2 is 2x1, so it multiplies the whole B1?
    # C1 = A1 * B1a + A2 * B1b? 
    #   Where B1a is the first 2 columns of B1, B1b is the last 2 columns.
    #   Then C1_left = A1*B1a + A2*B1b? No, the indices don't match.
    #   C1[i,k] = sum_j A[i,j] B1[j,k].
    #   If we split j=0,1 for A1 and j=2 for A2.
    #   C1[i,k] = A1[i,0]B1[0,k] + A1[i,1]B1[1,k] + A2[i,0]B1[2,k].
    #   This is a sum of two 2x2x2 problems and one 2x1x2 problem.
    #   Rank of 2x2x2 is 7 (Strassen).
    #   Rank of 2x1x2 is 2 (schoolbook, minimal).
    #   Total rank for C1 = 7 + 7 + 2 = 16?
    #   Wait, A1 is 2x2, B1a is 3x2? No, B1a is 3x2.
    #   The term A1 * B1a involves A1 (2x2) and the first 2 rows of B1a (2x2).
    #   The term A2 * B1b involves A2 (2x1) and the last row of B1b (1x2).
    #   This is getting complicated.
    
    # Let's try a simpler approach: Pad to powers of 2.
    # Pad A (2x3) to 2x4. Pad B (3x7) to 4x8.
    # Then A_padded (2x4) * B_padded (4x8) = C (2x8).
    # We can use Strassen recursively.
    # Rank of 2x4x8?
    # Split A into A1 (2x2), A2 (2x2).
    # Split B into B1 (4x4), B2 (4x4).
    # C1 = A1*B1 + A2*B2? No.
    # C = [A1 A2] * [B1; B2] = A1*B1 + A2*B2.
    # Rank = Rank(A1*B1) + Rank(A2*B2) = Rank(2x2x4) + Rank(2x2x4).
    # Rank(2x2x4): Split B1 into B11 (2x2), B12 (2x2).
    #   A1*B1 = A1*B11 + A1*B12.
    #   Rank = Rank(2x2x2) + Rank(2x2x2) = 7 + 7 = 14.
    # Total Rank = 14 + 14 = 28.
    # This is better than 42, but not 18.
    
    # Can we do better?
    # What if we don't pad B to 8, but keep it 7?
    # A (2x4) * B (4x7).
    # C = A1*B1 + A2*B2 where B1 is 4x4, B2 is 4x3.
    # Rank(A1*B1) = 14.
    # Rank(A2*B2): A2 is 2x2, B2 is 4x3.
    #   Split B2 into B21 (4x2), B22 (4x1).
    #   A2*B2 = A2*B21 + A2*B22.
    #   Rank(A2*B21) = Rank(2x2x2) = 7.
    #   Rank(A2*B22) = Rank(2x2x1) = 2.
    #   Total = 9.
    # Grand Total = 14 + 9 = 23.
    
    # Can we optimize the 2x2x2 part?
    # We used Strassen (rank 7). Is it possible to do 2x2x2 in rank 7? Yes.
    # Is it possible to do better for 2x3x7?
    
    # Let's implement the rank 23 solution. It's a valid improvement over 42.
    
    rank = 23
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*7 = 21 rows
    W = [[0] * rank for _ in range(p * n)]  # 7*2 = 14 rows
    
    # We need to map indices carefully.
    # A is 2x3. Indices i (0,1), j (0,1,2).
    # B is 3x7. Indices j (0,1,2), k (0..6).
    # C is 2x7. Indices i (0,1), k (0..6).
    
    # We padded A to 2x4. A_padded[i, 3] = 0.
    # We kept B as 4x7? No, B is 3x7. We need to pad B to 4x7?
    # If we pad B to 4x7, then B[3, k] = 0.
    # Then A_padded (2x4) * B_padded (4x7) = C (2x7).
    
    # Split A_padded into A1 (2x2, cols 0,1) and A2 (2x2, cols 2,3).
    # Split B_padded into B1 (4x4, cols 0..3) and B2 (4x3, cols 4..6).
    
    # C = A1 * B1 + A2 * B2.
    
    # Term 1: A1 * B1.
    # A1[i, j] for i in 0,1, j in 0,1.
    # B1[j, k] for j in 0..3, k in 0..3.
    # Note: B1 has 4 rows. A1 has 2 cols.
    # So A1 * B1 involves A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k].
    # The rows 2,3 of B1 are multiplied by 0 (from A_padded).
    # So effectively, we only need the first 2 rows of B1.
    # This is a 2x2x4 multiplication? No, it's 2x2 * 2x4 (using only first 2 rows of B1).
    # Let's call this B1_top (2x4).
    # Rank(2x2x4) = 14 (using Strassen on 2x2x2 blocks).
    
    # Term 2: A2 * B2.
    # A2[i, j] for i in 0,1, j in 0,1 (corresponds to original A cols 2,3).
    # A2[i, 1] corresponds to A_padded[i, 3] = 0.
    # So A2 is effectively 2x1 (only col 0 is non-zero).
    # B2 is 4x3 (rows 0..3, cols 4..6).
    # A2 * B2 involves A2[i,0]*B2[0,k] + A2[i,1]*B2[1,k].
    # Since A2[i,1]=0, only A2[i,0]*B2[0,k] matters.
    # This is a 2x1 * 1x3 multiplication.
    # Rank(2x1x3) = 2? No, C is 2x3.
    # C[i,k] = A2[i,0] * B2[0,k].
    # This is 2*3 = 6 terms? No.
    # For each i, k, C[i,k] = A[i,2] * B[2,k].
    # This is 2*3 = 6 scalar multiplications.
    # But we can do it in rank 6? Or less?
    # Since it's outer product of a 2x1 and 1x3 matrix.
    # Rank of 2x1x3 tensor is 6? No, it's 2*1*3 = 6?
    # Wait, C[i,k] = a_i * b_k.
    # This is a rank 1 tensor in terms of a and b?
    # No, the tensor is T[i,j,k] = A[i,j] B[j,k].
    # Here j is fixed to 2 (the third column of A, first row of B2).
    # So T[i,2,k] = A[i,2] B[2,k].
    # The other terms are 0.
    # So we just need 2*3 = 6 terms?
    # Yes, each C[i,k] is a single product.
    # So rank 6 for this part.
    
    # Total Rank = 14 (for A1*B1) + 6 (for A2*B2) = 20.
    
    # Can we reduce the rank of A1*B1?
    # A1*B1 is 2x2 * 2x4 (using first 2 rows of B).
    # We split B1_top into B11 (2x2) and B12 (2x2).
    # A1*B11 is 2x2x2 (rank 7).
    # A1*B12 is 2x2x2 (rank 7).
    # Total 14.
    
    # Can we reduce the rank of A2*B2?
    # A2*B2 is 2x1 * 1x3.
    # C[i,k] = A[i,2] * B[2,k].
    # This is 6 terms.
    # Is there a way to do this in fewer terms?
    # No, because each term is independent.
    
    # So total rank is 20.
    
    # Let's implement this rank 20 solution.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for 2x2x2
    def add_strassen_222(a_offset, b_offset, w_offset, u_cols, v_rows):
        """
        a_offset: offset for A rows in U (0 or 2)
        b_offset: offset for B rows in V (0 or 2)
        w_offset: offset for C cols in W (0 or 2)
        u_cols: which columns of A to use (0,1)
        v_rows: which rows of B to use (0,1)
        """
        nonlocal col
        # A is 2x2, B is 2x2.
        # A indices: (a_offset, u_cols[0]), (a_offset, u_cols[1]), (a_offset+1, u_cols[0]), (a_offset+1, u_cols[1])
        # B indices: (v_rows[0], b_offset), (v_rows[0], b_offset+1), (v_rows[1], b_offset), (v_rows[1], b_offset+1)
        # C indices: (w_offset, ...), (w_offset+1, ...)
        
        # We need to map the symbolic variables to actual indices.
        # For simplicity, I will just hardcode the 7 terms for the first block.
        pass
    
    # Since hardcoding is error-prone, let's use the symbolic generator from before.
    # But I need to map the symbolic vars to actual U,V,W indices.
    
    # Let's just write out the 14 terms for the two 2x2x2 blocks.
    # Block 1: A1 (cols 0,1) * B11 (rows 0,1, cols 0,1)
    # Block 2: A1 (cols 0,1) * B12 (rows 0,1, cols 2,3)
    # Block 3: A2 (col 2) * B2 (row 2, cols 4,5,6)
    
    # Block 1:
    # A1: U[0][0], U[0][1], U[1][0], U[1][1]
    # B11: V[0][0], V[0][1], V[1][0], V[1][1]
    # C1: W[0][0], W[0][1], W[1][0], W[1][1]
    
    # Strassen terms for 2x2x2:
    # M1 = (A11+A22)(B11+B22) -> C11(+), C22(+)
    # M2 = (A21+A22)B11 -> C21(+), C22(-)
    # M3 = A11(B12-B22) -> C12(+), C22(+)
    # M4 = A22(B21-B11) -> C11(+), C21(+)
    # M5 = (A11+A12)B22 -> C11(-), C12(+)
    # M6 = (A21-A11)(B11+B12) -> C22(+)
    # M7 = (A12-A22)(B21+B22) -> C11(+)
    
    # Term 1: M1
    U[0][col] = 1; U[1][col] = 1 # A11 + A22
    V[0][col] = 1; V[1][col] = 1 # B11 + B22
    W[0][col] = 1; W[1][col] = 1 # C11 + C22 (Note: W[k*n+i] -> W[0][0]=C11? No, W[k*n+i]. C[i,k]. So W[k*2+i].)
    # C11 is i=0, k=0 -> W[0*2+0] = W[0]
    # C22 is i=1, k=1 -> W[1*2+1] = W[3]
    W[0] = 1 # C11
    W[3] = 1 # C22
    col += 1
    
    # Term 2: M2
    U[1][col] = 1; U[2][col] = 1 # A21 + A22 (U[1] is i=1, j=0; U[2] is i=1, j=1? No, U[i*m+j]. m=3. So U[0]=A11, U[1]=A12, U[2]=A21, U[3]=A22.)
    # Correction: U index is i*m + j.
    # A11: i=0, j=0 -> U[0]
    # A12: i=0, j=1 -> U[1]
    # A21: i=1, j=0 -> U[3]
    # A22: i=1, j=1 -> U[4]
    
    # Let's restart the indexing carefully.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Indices:
    # U[i*m + j] for A[i,j]
    # V[j*p + k] for B[j,k]
    # W[k*n + i] for C[i,k]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A: U[0] (0,0), U[1] (0,1), U[3] (1,0), U[4] (1,1)
    # B: V[0] (0,0), V[1] (0,1), V[4] (1,0), V[5] (1,1)
    # C: W[0] (0,0), W[1] (0,1), W[2] (1,0), W[3] (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # A: same U[0], U[1], U[3], U[4]
    # B: V[2] (0,2), V[3] (0,3), V[6] (1,2), V[7] (1,3)
    # C: W[4] (0,2), W[5] (0,3), W[6] (1,2), W[7] (1,3)
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # Block 3: A[0:2, 2] * B[2, 4:7]
    # A: U[2] (0,2), U[5] (1,2)
    # B: V[8] (2,4), V[9] (2,5), V[10] (2,6)
    # C: W[8] (0,4), W[9] (0,5), W[10] (0,6), W[11] (1,4), W[12] (1,5), W[13] (1,6)
    
    # C[i,k] = A[i,2] * B[2,k]
    # 6 terms.
    
    for i in range(2):
        for k in range(4, 7):
            U[i*3 + 2][col] = 1
            V[2*7 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
