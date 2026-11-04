"""Strassen-based block decomposition for 2x3 * 3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    # We partition A (2x3) into blocks and B (3x7) into blocks.
    # Since n=2, m=3, p=7, we can use a mixed approach.
    # Let's try to use Strassen for a 2x2 block multiplication embedded in the larger product.
    #
    # Strategy:
    # A is 2x3. B is 3x7. C = A*B is 2x7.
    # We can view the multiplication as combining smaller multiplications.
    #
    # One effective approach: decompose the 3 intermediate dimension (m=3) into 2+1.
    # Let A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    # Let B = [B1; B2] where B1 is 2x7 and B2 is 1x7.
    # Then C = A1*B1 + A2*B2.
    #
    # A1*B1 is 2x2 * 2x7. We can use Strassen for the 2x2 part, treating the 7 columns as 7 separate 2x1 vectors,
    # or better, decompose the 7 columns into 2+2+2+1? No, let's keep it simple.
    #
    # Actually, let's use the known rank for 2x2 * 2x1 multiplication.
    # The tensor for 2x2 * 2x1 is isomorphic to 2x1 * 2x2? No.
    # Let's use the general bilinear map.
    #
    # Let's try a different partition: m=3 = 2+1.
    # C = A1 B1 + A2 B2
    # A1: 2x2, B1: 2x7 -> product C1: 2x7
    # A2: 2x1, B2: 1x7 -> product C2: 2x7
    #
    # For A1 B1 (2x2 * 2x7):
    # We can use Strassen's 7 multiplications for each 2x2 block of B1.
    # B1 has 7 columns. We can group them as 2+2+2+1? That's messy.
    # Alternatively, treat B1 as 7 vectors of size 2.
    # The multiplication of a 2x2 matrix by a 2x1 vector has rank 4 (schoolbook) or can it be lower?
    # Actually, the tensor <2,2,1> has rank 4. It is equivalent to <1,2,2>.
    # So A1 B1 requires 7 * 4 = 28 multiplications? No, Strassen applies to 2x2 * 2x2.
    # If we treat B1 as four 2x2 blocks plus one 2x1, we can use Strassen for the 2x2 parts.
    # B1 is 2x7. Let B1 = [B11 B12 B13 B14] where B11,B12,B13 are 2x2 and B14 is 2x1.
    # C1 = A1 B11 + A1 B12 + A1 B13 + A1 B14.
    # This doesn't combine well because A1 is shared.
    #
    # Let's use the standard Strassen decomposition for the whole thing if possible.
    # But dimensions are not powers of 2.
    #
    # Let's try a direct construction using known low-rank decompositions for small tensors.
    # The tensor is <2,3,7>.
    # We can use the decomposition:
    # <2,3,7> <= <2,2,4> + <2,1,3> ? No.
    #
    # Let's use the fact that <2,3,7> can be embedded in a larger tensor or decomposed.
    # A = [a11 a12 a13; a21 a22 a23]
    # B = [b11 ... b17; b21 ... b27; b31 ... b37]
    # C = [c11 ... c17; c21 ... c27]
    #
    # c1j = a11*b1j + a12*b2j + a13*b3j
    # c2j = a21*b1j + a22*b2j + a23*b3j
    #
    # This is 14 output entries, each a sum of 3 products. Total 42 products.
    # We want to reduce this.
    #
    # Let's use the decomposition of the 3x7 part.
    # We can write B as a sum of 7 rank-1 matrices? No.
    #
    # Let's try to use the rank of <2,3,3> which is known to be 8 (Landsberg).
    # <2,3,3> <= <2,2,2> + <2,1,1> ? No.
    # Actually, <2,3,3> has rank 8.
    # Then <2,3,7> can be seen as <2,3,3> + <2,3,3> + <2,3,1>.
    # Rank(<2,3,3>) = 8.
    # Rank(<2,3,1>) = 3 (since it's 2*3*1, but actually <2,3,1> is isomorphic to <1,3,2> which has rank 3? No, <2,3,1> has rank 6? Wait.
    # The tensor <n,m,p> for p=1 is just a bilinear map from n*m to 1? No, it's n x m x 1.
    # The rank of <n,m,1> is n*m. So <2,3,1> has rank 6.
    # So total rank would be 8 + 8 + 6 = 22.
    #
    # Let's verify the decomposition:
    # We split the 7 columns of B into 3+3+1.
    # B = [B1 B2 B3] where B1 is 3x3, B2 is 3x3, B3 is 3x1.
    # C = A B1 + A B2 + A B3.
    # Each term A Bi is a 2x3 * 3x3 or 2x3 * 3x1 product.
    # The tensor for 2x3 * 3x3 is <2,3,3>. Rank 8.
    # The tensor for 2x3 * 3x1 is <2,3,1>. Rank 6.
    # Total rank 8+8+6 = 22.
    #
    # We need to explicitly construct the U, V, W matrices for this decomposition.
    #
    # Let's define the factors for <2,3,3> with rank 8.
    # And for <2,3,1> with rank 6.
    #
    # For <2,3,1>:
    # A is 2x3, B is 3x1, C is 2x1.
    # c1 = a11*b1 + a12*b2 + a13*b3
    # c2 = a21*b1 + a22*b2 + a23*b3
    # This is just 6 independent products.
    # U: 6x6, V: 3x6, W: 2x6.
    # U[i, j] = 1 if i corresponds to (row, col) in A.
    # V[j, k] = 1 if j corresponds to (col, row) in B.
    # W[k, l] = 1 if k corresponds to (row, col) in C.
    #
    # For <2,3,3> with rank 8:
    # This is more complex. We need a known rank-8 decomposition.
    # One such decomposition is given by Landsberg.
    # Let's use a simple construction or look up the coefficients.
    #
    # Alternatively, we can use the decomposition:
    # <2,3,3> <= <2,2,2> + <2,1,1> + <1,2,1> + ...?
    #
    # Let's use a different approach: use the rank-7 decomposition for <2,2,2> (Strassen)
    # and add the remaining terms.
    # <2,3,3> can be viewed as <2,2,2> + <2,1,2> + <2,2,1> + <2,1,1>?
    # This is getting complicated.
    #
    # Let's try a simpler decomposition:
    # <2,3,7> <= 7 * <2,3,1> = 7 * 6 = 42. (Schoolbook)
    #
    # How about <2,3,7> <= <2,2,4> + <2,1,3>?
    # Rank(<2,2,4>) = ? <2,2,4> is isomorphic to <2,4,2>.
    # Rank(<2,4,2>) is at most 2*4*2 = 16, but can be lower.
    # Actually, <2,2,4> can be decomposed using Strassen for each 2x2 block of the 4 columns.
    # So <2,2,4> <= 2 * <2,2,2> = 2 * 7 = 14.
    # <2,1,3> has rank 3 (since it's 2*1*3, but wait, <2,1,3> is isomorphic to <1,3,2> which has rank 3? No, rank is 2*1*3=6? No.
    # The tensor <n,m,p> has rank at most n*m*p. For <2,1,3>, the rank is 3?
    # Let's check: c1 = a1*b1 + a1*b2 + a1*b3? No.
    # A is 2x1, B is 1x3, C is 2x3.
    # c11 = a1*b1, c12 = a1*b2, c13 = a1*b3
    # c21 = a2*b1, c22 = a2*b2, c23 = a2*b3
    # This is 6 products. So rank is 6.
    #
    # So <2,2,4> + <2,1,3> = 14 + 6 = 20.
    # This is better than 42.
    #
    # Let's use this decomposition:
    # Split B (3x7) into B1 (2x7) and B2 (1x7)? No, B is 3x7.
    # Split the middle dimension m=3 into 2+1.
    # A = [A1 A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x7, B2 is 1x7.
    # C = A1 B1 + A2 B2.
    #
    # A1 B1 is 2x2 * 2x7.
    # We can decompose 2x2 * 2x7 as 7 * (2x2 * 2x1).
    # Rank of 2x2 * 2x1 is 4. So 7*4=28.
    # But we can do better by grouping columns.
    # 2x2 * 2x7 can be seen as 2x2 * (2x2 + 2x2 + 2x2 + 2x1).
    # So 3 * (2x2 * 2x2) + 1 * (2x2 * 2x1).
    # Rank(2x2 * 2x2) = 7 (Strassen).
    # Rank(2x2 * 2x1) = 4.
    # Total for A1 B1: 3*7 + 4 = 25.
    #
    # A2 B2 is 2x1 * 1x7.
    # This is 2x1 * (1x2 + 1x2 + 1x2 + 1x1).
    # Rank(2x1 * 1x2) = 2 (since it's just scaling).
    # Rank(2x1 * 1x1) = 2.
    # Total for A2 B2: 3*2 + 2 = 8.
    #
    # Total rank: 25 + 8 = 33.
    #
    # This is still better than 42. Let's try to optimize further.
    #
    # Actually, for 2x2 * 2x1, the rank is 4, but can we do better?
    # The tensor <2,2,1> is isomorphic to <2,1,2>.
    # The rank of <2,1,2> is 2? No, it's 4.
    # Wait, <2,1,2>: A is 2x1, B is 1x2, C is 2x2.
    # c11 = a1*b1, c12 = a1*b2, c21 = a2*b1, c22 = a2*b2.
    # This is 4 products. So rank is 4.
    #
    # So 33 is a good target. Let's implement this.
    #
    # We need to construct U, V, W for the whole decomposition.
    #
    # Let's define the factors for each part.
    #
    # Part 1: A1 B1 where A1 is 2x2, B1 is 2x7.
    # We split B1 into B11 (2x2), B12 (2x2), B13 (2x2), B14 (2x1).
    # C1 = A1 B11 + A1 B12 + A1 B13 + A1 B14.
    #
    # For each 2x2 * 2x2, we use Strassen (rank 7).
    # For 2x2 * 2x1, we use schoolbook (rank 4).
    #
    # Part 2: A2 B2 where A2 is 2x1, B2 is 1x7.
    # We split B2 into B21 (1x2), B22 (1x2), B23 (1x2), B24 (1x1).
    # C2 = A2 B21 + A2 B22 + A2 B23 + A2 B24.
    #
    # For each 2x1 * 1x2, we use schoolbook (rank 2).
    # For 2x1 * 1x1, we use schoolbook (rank 2).
    #
    # Total rank: 3*7 + 4 + 3*2 + 2 = 21 + 4 + 6 + 2 = 33.
    #
    # Let's implement this.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6
    V = [[0] * rank for _ in range(m * p)]  # 3*7 = 21
    W = [[0] * rank for _ in range(p * n)]  # 7*2 = 14
    
    col = 0
    
    # Part 1: A1 B1
    # A1 is rows 0,1 of A (cols 0,1)
    # B1 is rows 0,1 of B (cols 0-6)
    
    # B11: cols 0,1 of B1
    # Strassen for 2x2 * 2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # We need to map this to our U, V, W.
    # U indices: i*m+j
    # V indices: j*p+k
    # W indices: k*n+i
    
    # Let's write a helper to add a Strassen block.
    def add_strassen_2x2_2x2(u_offset, v_offset, w_offset, start_col):
        # u_offset: index in U for A1 (2x2)
        # v_offset: index in V for B11 (2x2)
        # w_offset: index in W for C1 (2x2)
        # start_col: column index in U, V, W
        nonlocal col
        # M1
        U[u_offset][start_col] += 1  # a11
        U[u_offset+2][start_col] += 1  # a22 (u_offset+2 is a22)
        V[v_offset][start_col] += 1  # b11
        V[v_offset+2][start_col] += 1  # b22 (v_offset+2 is b22)
        W[w_offset][start_col] += 1  # c11
        W[w_offset+2][start_col] += 1  # c22 (w_offset+2 is c22)
        col += 1
        
        # M2
        U[u_offset+1][start_col] += 1  # a21
        U[u_offset+2][start_col] += 1  # a22
        V[v_offset][start_col] += 1  # b11
        W[w_offset+1][start_col] += 1  # c21 (w_offset+1 is c21)
        col += 1
        
        # M3
        U[u_offset][start_col] += 1  # a11
        V[v_offset+1][start_col] += 1  # b12
        V[v_offset+2][start_col] -= 1  # b22
        W[w_offset][start_col] += 1  # c11
        W[w_offset+1][start_col] += 1  # c12 (w_offset+1 is c12? No, c12 is w_offset+1)
        # Wait, W indices: k*n+i.
        # C1 is 2x2. w_offset is c11, w_offset+1 is c12, w_offset+2 is c21, w_offset+3 is c22.
        # Let's correct the W indices.
        # Actually, W is 14x33. The indices are k*n+i.
        # For C1 (2x2), the indices are:
        # c11: k=0, i=0 -> 0*2+0=0
        # c12: k=1, i=0 -> 1*2+0=2
        # c21: k=0, i=1 -> 0*2+1=1
        # c22: k=1, i=1 -> 1*2+1=3
        # So w_offset+0, w_offset+2, w_offset+1, w_offset+3.
        # Let's redefine the W offsets.
        # This is getting messy. Let's use a more systematic approach.
        pass
    
    # Given the complexity, let's use a simpler decomposition that is easier to implement.
    # We'll use the 22-rank decomposition from earlier:
    # <2,3,7> <= <2,3,3> + <2,3,3> + <2,3,1>
    # Rank 8 + 8 + 6 = 22.
    #
    # We need a rank-8 decomposition for <2,3,3>.
    # One known decomposition is:
    # Let A be 2x3, B be 3x3, C be 2x3.
    # We can use the following 8 terms:
    # This is complex to hardcode.
    #
    # Let's stick to the 33-rank decomposition which is easier to construct.
    #
    # Let's implement the 33-rank decomposition carefully.
    
    # Reset
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: A1 B1 (2x2 * 2x7)
    # A1: U indices 0,1,2,3 (a11,a12,a21,a22)
    # B1: V indices 0..13 (b11..b17, b21..b27)
    # C1: W indices 0..13 (c11..c17, c21..c27)
    
    # B11: V indices 0,1,2,3 (b11,b12,b21,b22)
    # B12: V indices 4,5,6,7 (b13,b14,b23,b24)
    # B13: V indices 8,9,10,11 (b15,b16,b25,b26)
    # B14: V indices 12,13 (b17,b27)
    
    # C11: W indices 0,1 (c11,c12)
    # C12: W indices 2,3 (c13,c14)
    # C13: W indices 4,5 (c15,c16)
    # C14: W indices 6,7 (c17,c21) -> Wait, C is 2x7.
    # W indices: k*n+i.
    # c11: 0*2+0=0
    # c12: 1*2+0=2
    # c13: 2*2+0=4
    # c14: 3*2+0=6
    # c15: 4*2+0=8
    # c16: 5*2+0=10
    # c17: 6*2+0=12
    # c21: 0*2+1=1
    # c22: 1*2+1=3
    # c23: 2*2+1=5
    # c24: 3*2+1=7
    # c25: 4*2+1=9
    # c26: 5*2+1=11
    # c27: 6*2+1=13
    
    # So C1 (first 4 cols of C) uses W indices 0,1,2,3,4,5,6,7.
    # C2 (next 3 cols of C) uses W indices 8,9,10,11,12,13.
    
    # Let's define a function to add a 2x2*2x2 Strassen block.
    def add_strassen(u_off, v_off, w_off, c):
        nonlocal col
        # M1
        U[u_off][c] += 1; U[u_off+2][c] += 1
        V[v_off][c] += 1; V[v_off+2][c] += 1
        W[w_off][c] += 1; W[w_off+2][c] += 1
        col += 1
        # M2
        U[u_off+1][c] += 1; U[u_off+2][c] += 1
        V[v_off][c] += 1
        W[w_off+1][c] += 1
        col += 1
        # M3
        U[u_off][c] += 1
        V[v_off+1][c] += 1; V[v_off+2][c] -= 1
        W[w_off][c] += 1; W[w_off+1][c] += 1
        col += 1
        # M4
        U[u_off+2][c] += 1
        V[v_off+1][c] += 1; V[v_off][c] -= 1
        W[w_off+1][c] += 1; W[w_off][c] += 1
        col += 1
        # M5
        U[u_off][c] += 1; U[u_off+1][c] += 1
        V[v_off+2][c] += 1
        W[w_off][c] -= 1; W[w_off+2][c] += 1
        col += 1
        # M6
        U[u_off+1][c] -= 1; U[u_off][c] += 1
        V[v_off][c] += 1; V[v_off+1][c] += 1
        W[w_off+2][c] += 1
        col += 1
        # M7
        U[u_off+1][c] -= 1; U[u_off+2][c] += 1
        V[v_off+1][c] += 1; V[v_off+2][c] += 1
        W[w_off][c] += 1
        col += 1
    
    # Add Strassen for B11, B12, B13
    add_strassen(0, 0, 0, col)
    add_strassen(0, 4, 4, col)
    add_strassen(0, 8, 8, col)
    
    # Add schoolbook for B14 (2x2 * 2x1)
    # c17 = a11*b17 + a12*b27
    # c27 = a21*b17 + a22*b27
    U[0][col] += 1; V[12][col] += 1; W[12][col] += 1; col += 1
    U[1][col] += 1; V[13][col] += 1; W[12][col] += 1; col += 1
    U[2][col] += 1; V[12][col] += 1; W[13][col] += 1; col += 1
    U[3][col] += 1; V[13][col] += 1; W[13][col] += 1; col += 1
    
    # Part 2: A2 B2 (2x1 * 1x7)
    # A2: U indices 4,5 (a13,a23)
    # B2: V indices 14..20 (b31..b37)
    # C2: W indices 14..20? No, C is 2x7.
    # C2 corresponds to the 3rd row of B, which affects all columns of C.
    # c1j += a13*b3j
    # c2j += a23*b3j
    # So W indices for C2 are the same as C1, but we are adding to them.
    # Wait, the decomposition is C = A1 B1 + A2 B2.
    # So the W indices for A2 B2 are the same as for A1 B1, but we are adding new terms.
    # This means we need to add to the existing W entries.
    #
    # Let's add the terms for A2 B2.
    # B21: V indices 14,15 (b31,b32)
    # B22: V indices 16,17 (b33,b34)
    # B23: V indices 18,19 (b35,b36)
    # B24: V index 20 (b37)
    
    # For each 1x2 block, we add 2 terms.
    # c11 += a13*b31, c12 += a13*b32
    # c21 += a23*b31, c22 += a23*b32
    U[4][col] += 1; V[14][col] += 1; W[0][col] += 1; col += 1
    U[4][col] += 1; V[15][col] += 1; W[2][col] += 1; col += 1
    U[5][col] += 1; V[14][col] += 1; W[1][col] += 1; col += 1
    U[5][col] += 1; V[15][col] += 1; W[3][col] += 1; col += 1
    
    U[4][col] += 1; V[16][col] += 1; W[4][col] += 1; col += 1
    U[4][col] += 1; V[17][col] += 1; W[6][col] += 1; col += 1
    U[5][col] += 1; V[16][col] += 1; W[5][col] += 1; col += 1
    U[5][col] += 1; V[17][col] += 1; W[7][col] += 1; col += 1
    
    U[4][col] += 1; V[18][col] += 1; W[8][col] += 1; col += 1
    U[4][col] += 1; V[19][col] += 1; W[10][col] += 1; col += 1
    U[5][col] += 1; V[18][col] += 1; W[9][col] += 1; col += 1
    U[5][col] += 1; V[19][col] += 1; W[11][col] += 1; col += 1
    
    U[4][col] += 1; V[20][col] += 1; W[12][col] += 1; col += 1
    U[5][col] += 1; V[20][col] += 1; W[13][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
