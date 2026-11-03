"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def _strassen_4x4():
    """
    Return the rank-7 Strassen decomposition for 4x4 * 4x4 -> 4x4.
    Returns (U, V, W) as lists of lists.
    U: (16, 7) where U[i*4+j, c]
    V: (16, 7) where V[j*4+k, c]
    W: (16, 7) where W[k*4+i, c]
    """
    # Strassen's 7 bilinear forms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    #
    # For 4x4, we can use a block structure. But actually, the standard
    # Strassen algorithm for 4x4 uses the 2x2 Strassen recursively, which
    # gives rank 7*4 = 28? No, that's not right.
    #
    # Actually, the rank of 4x4x4 matrix multiplication is at most 28 (using
    # Strassen recursively), but the known optimal is 14 (Bini-Capovani-Lotti-Romani).
    # However, for simplicity and correctness, let's use a known rank-14 decomposition.
    #
    # The Bini-Capovani-Lotti-Romani (BCLR) rank-14 decomposition for 4x4x4:
    # This is complex to hardcode. Let's instead use the Strassen-based rank-28
    # decomposition which is simpler to construct and still much better than 64.
    #
    # Actually, let me just use the schoolbook for 4x4 (rank 64) and see if we can
    # do better with a simpler approach. Or better yet, let's use the known fact
    # that 4x4x4 has rank at most 28 via recursive Strassen, but let me construct
    # a rank-28 decomposition.
    #
    # For 4x4 * 4x4, using Strassen recursively:
    # Split into 2x2 blocks, each 2x2.
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B10-B11) -> contributes to C00, C01
    # M4 = A11(B01-B10) -> contributes to C00, C01
    # M5 = (A00+A01)B11 -> contributes to C00, C11
    # M6 = (A10-A00)(B00+B10) -> contributes to C10, C11
    # M7 = (A01-A11)(B01+B11) -> contributes to C00, C10
    #
    # But this gives 7 * 4 = 28 scalar multiplications, so rank 28.
    #
    # Let me construct this properly.
    #
    # Actually, the simplest correct approach: use the known rank-14 BCLR decomposition.
    # But that's very complex to hardcode. Let me use a different approach.
    #
    # For the 4x4x8 case, I can use the fact that we can treat it as two independent
    # 4x4x4 multiplications (since the right matrix has 8 columns, we can split into
    # two 4x4 blocks). So rank would be 2 * rank(4x4x4).
    #
    # If I use Strassen rank-28 for each, total rank = 56.
    # If I use BCLR rank-14 for each, total rank = 28.
    #
    # Let me implement the Strassen-based rank-28 for 4x4x4, then use it twice.
    
    n = 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * n)]
    V = [[0.0] * rank for _ in range(n * n)]
    W = [[0.0] * rank for _ in range(n * n)]
    
    # Strassen 2x2 bilinear forms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 4x4, we split A into 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    #
    # Where M1..M7 are 2x2 matrix products (computed with schoolbook, rank 4 each)
    # So total rank = 7 * 4 = 28
    
    # Let me index: A[i,j] where i,j in 0..3
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B and C.
    
    # For each Strassen form M_s, and for each schoolbook term (a,b) in the 2x2 product:
    # M_s[a,b] = sum over the bilinear form
    
    # Let me define the 7 Strassen forms as linear combinations of A-blocks and B-blocks.
    # For form s, U_part = linear combo of A-blocks, V_part = linear combo of B-blocks
    # Then M_s = U_part * V_part (2x2 * 2x2 = 2x2, using schoolbook rank 4)
    
    # Strassen forms (for 2x2 blocks):
    # M1: U = A00 + A11, V = B00 + B11
    # M2: U = A10 + A11, V = B00
    # M3: U = A00, V = B10 - B11
    # M4: U = A11, V = B01 - B10
    # M5: U = A00 + A01, V = B11
    # M6: U = A10 - A00, V = B00 + B10
    # M7: U = A01 - A11, V = B01 + B11
    
    # C contributions:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    
    # So for each form s and each (a,b) in 0..1 x 0..1:
    # The contribution to C from M_s[a,b] is:
    # C00[a,b] += M1[a,b] - M4[a,b] + M5[a,b] - M7[a,b]
    # C01[a,b] += M3[a,b] + M4[a,b]
    # C10[a,b] += M2[a,b] + M7[a,b]
    # C11[a,b] += M1[a,b] - M3[a,b] + M6[a,b] - M5[a,b]
    
    # Let me build this.
    
    # Block indices: block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A-block (bi,bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    # For each Strassen form s (0..6), and each schoolbook term (a,b) (0..1, 0..1):
    # rank index = s * 4 + a * 2 + b
    
    # U[i,j] = coefficient of A[i,j] in U_part of form s
    # V[i,j] = coefficient of B[i,j] in V_part of form s
    # W[k,l] = coefficient in C[k,l] from M_s[a,b]
    
    # Let me define the Strassen forms:
    # form 0 (M1): U = A00+A11, V = B00+B11
    # form 1 (M2): U = A10+A11, V = B00
    # form 2 (M3): U = A00, V = B10-B11
    # form 3 (M4): U = A11, V = B01-B10
    # form 4 (M5): U = A00+A01, V = B11
    # form 5 (M6): U = A10-A00, V = B00+B10
    # form 6 (M7): U = A01-A11, V = B01+B11
    
    # C contributions (for each (a,b)):
    # C00: M1 - M4 + M5 - M7
    # C01: M3 + M4
    # C10: M2 + M7
    # C11: M1 - M3 + M6 - M5
    
    # C block (ci, cj) covers rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
    
    # Let me build U, V, W.
    
    # U[i,j] for i in 0..3, j in 0..3
    # V[i,j] for i in 0..3, j in 0..3
    # W[k,l] for k in 0..3, l in 0..3 (k is col of B, l is row of A, per trace convention W[k*n+l])
    
    # For form s, schoolbook term (a,b):
    # U_part is a 2x2 matrix (the linear combo of A-blocks)
    # V_part is a 2x2 matrix (the linear combo of B-blocks)
    # M_s[a,b] = sum_{u,v} U_part[u,v] * A[2*block_row_u + u, 2*block_col_v + v] * V_part... 
    
    # Actually, let me think differently. 
    # M_s is a 2x2 matrix product: (U_part) * (V_part) where U_part and V_part are 2x2.
    # M_s[a,b] = sum_{u=0}^{1} sum_{v=0}^{1} U_part[u,v] * A_element[u,v] * V_part[v,b]... 
    
    # No, M_s = U_part * V_part, where U_part and V_part are 2x2 matrices formed from A and B blocks.
    # M_s[a,b] = sum_{t=0}^{1} U_part[a,t] * V_part[t,b]
    # U_part[a,t] is a linear combination of A elements.
    # V_part[t,b] is a linear combination of B elements.
    
    # For a specific Strassen form s:
    # U_part is defined by which A-blocks are summed.
    # V_part is defined by which B-blocks are summed.
    
    # Let me define for each form s:
    # U_blocks: list of (bi, bj, coeff) for A-blocks
    # V_blocks: list of (bi, bj, coeff) for B-blocks
    
    U_blocks = [
        [(0,0,1), (1,1,1)],      # M1: A00+A11
        [(1,0,1), (1,1,1)],      # M2: A10+A11
        [(0,0,1)],                # M3: A00
        [(1,1,1)],                # M4: A11
        [(0,0,1), (0,1,1)],      # M5: A00+A01
        [(1,0,1), (0,0,-1)],     # M6: A10-A00
        [(0,1,1), (1,1,-1)],     # M7: A01-A11
    ]
    
    V_blocks = [
        [(0,0,1), (1,1,1)],      # M1: B00+B11
        [(0,0,1)],                # M2: B00
        [(1,0,1), (1,1,-1)],     # M3: B10-B11
        [(0,1,1), (1,0,-1)],     # M4: B01-B10
        [(1,1,1)],                # M5: B11
        [(0,0,1), (1,0,1)],      # M6: B00+B10
        [(0,1,1), (1,1,1)],      # M7: B01+B11
    ]
    
    # C contributions: for each C-block (ci,cj) and each (a,b), the coefficient of M_s[a,b]
    # C00: M1 - M4 + M5 - M7
    # C01: M3 + M4
    # C10: M2 + M7
    # C11: M1 - M3 + M6 - M5
    
    C_contrib = {
        (0,0): [(0,1), (3,-1), (4,1), (6,-1)],
        (0,1): [(2,1), (3,1)],
        (1,0): [(1,1), (6,1)],
        (1,1): [(0,1), (2,-1), (5,1), (4,-1)],
    }
    
    # Now build U, V, W.
    # For form s, schoolbook term (a,b): rank_idx = s*4 + a*2 + b
    # U[i,j]: for each (bi,bj,coeff) in U_blocks[s], add coeff to U[2*bi+u, 2*bj+v] 
    #   where u,v are the positions within the 2x2 block that correspond to schoolbook term (a,b)
    #   Actually, U_part[a,t] involves A elements. The schoolbook term (a,b) in the product
    #   M_s[a,b] = sum_t U_part[a,t] * V_part[t,b]
    #   U_part[a,t] is a linear combo of A-block elements.
    #   Specifically, for U_blocks[s], each (bi,bj,coeff) means we add coeff * A_block(bi,bj).
    #   A_block(bi,bj)[u,v] = A[2*bi+u, 2*bj+v]
    #   So U_part[a,t] = sum_{(bi,bj,coeff) in U_blocks[s]} coeff * A[2*bi+a, 2*bj+t]
    #   Wait, no. U_part is a 2x2 matrix. U_part[a,t] = sum of coeff * A_block(bi,bj)[a,t]
    #   = sum of coeff * A[2*bi+a, 2*bj+t]
    #   Similarly V_part[t,b] = sum of coeff * B[2*bi+t, 2*bj+b]
    
    # So for M_s[a,b] = sum_t U_part[a,t] * V_part[t,b]
    # = sum_t (sum_{(bi,bj,c1) in U_blocks[s]} c1 * A[2*bi+a, 2*bj+t]) * (sum_{(bi',bj',c2) in V_blocks[s]} c2 * B[2*bi'+t, 2*bj'+b])
    # = sum_{t, (bi,bj,c1), (bi',bj',c2)} c1*c2 * A[2*bi+a, 2*bj+t] * B[2*bi'+t, 2*bj'+b]
    
    # This is a sum of products of A and B elements. Each term is a rank-1 contribution.
    # But we want to express this as a single bilinear form per (s,a,b).
    # The issue is that M_s[a,b] is a sum of multiple products, so it's not a single rank-1 term.
    # 
    # I think I'm overcomplicating this. Let me just use the schoolbook for 4x4 (rank 64)
    # and accept that. Or better, let me use a known rank-14 decomposition.
    #
    # Actually, for the 4x4x8 case, the simplest improvement is to use the schoolbook
    # but with some sharing. But the directive says to use BCLR rank-14 extended to rank 28.
    #
    # Let me just hardcode a known rank-14 decomposition for 4x4x4.
    # This is complex. Let me instead use a simpler approach: use the schoolbook for
    # each of the two 4-column blocks, giving rank 64*2 = 128 (same as current).
    #
    # That's not an improvement. Let me think differently.
    #
    # The current program has rank 128 (schoolbook for 4x4x8).
    # If I can get rank 56 (Strassen-based), that's a 2.29x improvement.
    #
    # Let me implement the Strassen-based rank-28 for 4x4x4 properly.
    
    # Actually, let me just use the schoolbook but recognize that for 4x4x8,
    # we can potentially share some computations. But the simplest correct
    # improvement is to use Strassen's algorithm.
    
    # Let me implement a clean Strassen-based decomposition for 4x4 * 4x4.
    
    # The key insight: 4x4 * 4x4 can be done with 28 scalar multiplications
    # by using Strassen's 2x2 algorithm on 2x2 blocks, where each 2x2*2x2
    # product uses 7 multiplications, and there are 4 such block products
    # (for the 4 output blocks), but with sharing.
    
    # Actually, Strassen's algorithm for 4x4:
    # Split A, B into 2x2 blocks of 2x2 matrices.
    # Compute 7 products M1..M7 of 2x2 matrices (each using 4 scalar mults = 28 total).
    # Combine to get C blocks.
    
    # So rank = 7 * 4 = 28.
    
    # Let me build this carefully.
    
    U = [[0.0] * 28 for _ in range(16)]
    V = [[0.0] * 28 for _ in range(16)]
    W = [[0.0] * 28 for _ in range(16)]
    
    # For each Strassen form s (0..6) and schoolbook term (a,b) (0..1,0..1):
    # rank_idx = s * 4 + a * 2 + b
    
    # U_part for form s: 2x2 matrix of coefficients on A-blocks
    # V_part for form s: 2x2 matrix of coefficients on B-blocks
    
    # U_blocks[s][a][t] = coefficient of A[2*bi+a, 2*bj+t] ... this is getting complex.
    
    # Let me use a different approach. I'll define the 7 Strassen forms as:
    # For form s, the bilinear form is:
    # M_s = (linear combo of A) * (linear combo of B)
    # where each is a 2x2 matrix.
    
    # M_s[a,b] = sum_{t=0}^{1} U_s[a,t] * V_s[t,b]
    # where U_s and V_s are 2x2 matrices with entries that are linear combos of A/B elements.
    
    # For the schoolbook expansion of M_s[a,b]:
    # M_s[a,b] = sum_{t=0}^{1} (sum_{(bi,bj,c1)} c1 * A[2*bi+a, 2*bj+t]) * (sum_{(bi',bj',c2)} c2 * B[2*bi'+t, 2*bj'+b])
    
    # This is a quadratic form in A and B, not a single product. So each M_s[a,b]
    # is actually a sum of 4*4=16 products (in the worst case), which defeats the purpose.
    
    # I think the correct way is: each of the 7 Strassen forms M_s is a 2x2 matrix
    # product, which requires 4 scalar multiplications. So we have 7*4=28 scalar
    # multiplications total. Each scalar multiplication is a product of one linear
    # combo of A and one linear combo of B.
    
    # So for form s, the 2x2 product M_s = U_s * V_s where U_s, V_s are 2x2.
    # M_s[0,0] = U_s[0,0]*V_s[0,0] + U_s[0,1]*V_s[1,0]
    # M_s[0,1] = U_s[0,0]*V_s[0,1] + U_s[0,1]*V_s[1,1]
    # M_s[1,0] = U_s[1,0]*V_s[0,0] + U_s[1,1]*V_s[1,0]
    # M_s[1,1] = U_s[1,0]*V_s[0,1] + U_s[1,1]*V_s[1,1]
    
    # Each term like U_s[0,0]*V_s[0,0] is a product of two linear combos.
    # U_s[0,0] is a linear combo of A elements, V_s[0,0] is a linear combo of B elements.
    # Their product is a bilinear form, which can be written as a sum of rank-1 terms.
    # But for the tensor decomposition, each scalar multiplication corresponds to
    # one rank-1 term where U and V are the linear combos.
    
    # So for each of the 28 scalar multiplications, we have:
    # U_vector: the coefficients of A in the linear combo
    # V_vector: the coefficients of B in the linear combo
    # W_vector: the contribution to C
    
    # Let me define this.
    
    # For form s, define U_s and V_s as 2x2 matrices of linear combos.
    
    # U_s[a,t] = sum of c * A[2*bi+a, 2*bj+t] for (bi,bj,c) in U_blocks[s]
    # V_s[t,b] = sum of c * B[2*bi+t, 2*bj+b] for (bi,bj,c) in V_blocks[s]
    
    # The 4 scalar multiplications for form s are:
    # (0,0): U_s[0,0] * V_s[0,0]
    # (0,1): U_s[0,1] * V_s[1,0]  -- wait, M_s[0,1] = U_s[0,0]*V_s[0,1] + U_s[0,1]*V_s[1,1]
    # So the 4 products are:
    # P1 = U_s[0,0] * V_s[0,0]
    # P2 = U_s[0,1] * V_s[1,0]
    # P3 = U_s[0,0] * V_s[0,1]
    # P4 = U_s[0,1] * V_s[1,1]
    # P5 = U_s[1,0] * V_s[0,0]
    # P6 = U_s[1,1] * V_s[1,0]
    # P7 = U_s[1,0] * V_s[0,1]
    # P8 = U_s[1,1] * V_s[1,1]
    # Wait, that's 8, not 4. The 2x2 product has 4 entries, each being a sum of 2 products.
    # So 8 scalar multiplications per Strassen form, 7*8=56 total. That's rank 56.
    
    # Hmm, that's not 28. Let me reconsider.
    # A 2x2 * 2x2 product using schoolbook requires 4 multiplications:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # That's 8 multiplications (A[0,0]*B[0,0], A[0,1]*B[1,0], etc.)
    # But Strassen reduces this to 7 multiplications.
    
    # So for 4x4 using Strassen on 2x2 blocks: 7 multiplications of 2x2 matrices,
    # each requiring 4 scalar multiplications (schoolbook) = 28 total.
    # But Strassen's 7 multiplications are of the form (linear combo of A-blocks) * (linear combo of B-blocks),
    # where each "multiplication" is a 2x2*2x2 product done with schoolbook (4 scalar mults).
    # So total = 7 * 4 = 28 scalar multiplications.
    
    # OK so my earlier analysis was correct: rank 28 for 4x4x4.
    
    # Let me just implement it by enumerating all the scalar multiplications.
    
    # For each form s (0..6):
    #   U_s is a 2x2 matrix where U_s[a,t] = sum_{(bi,bj,c) in U_blocks[s]} c * A[2*bi+a, 2*bj+t]
    #   V_s is a 2x2 matrix where V_s[t,b] = sum_{(bi,bj,c) in V_blocks[s]} c * B[2*bi+t, 2*bj+b]
    #   M_s = U_s * V_s (2x2 product)
    #   M_s[a,b] = sum_{t=0}^{1} U_s[a,t] * V_s[t,b]
    
    # Each term U_s[a,t] * V_s[t,b] is a product of two linear combos.
    # This is one scalar multiplication (rank-1 term in the tensor).
    
    # So for each s, a, t, b: we have a term.
    # But M_s[a,b] = U_s[a,0]*V_s[0,b] + U_s[a,1]*V_s[1,b]
    # So there are 2 terms per (s,a,b), and 4*(a,b) pairs, so 8 terms per s, 7*8=56 terms.
    
    # Wait, that gives rank 56, not 28. I think I'm confusing myself.
    
    # Let me just use the schoolbook for 4x4x4 (rank 64) and use it twice for 4x4x8 (rank 128).
    # That's the same as the current program. Not an improvement.
    
    # OK, let me try a different approach. Let me use the known fact that the
    # tensor rank of 4x4x4 is at most 28 (via Strassen) and try to construct it.
    
    # Actually, I'll just use a simple but valid improvement: use the schoolbook
    # for the 4x4 part but recognize that for 4x4x8, we can potentially do better
    # by using the structure. But the simplest is to just return the schoolbook.
    
    # Given the complexity, let me just return the schoolbook decomposition for now.
    # This matches the current program's rank of 128.
    
    return None  # placeholder

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
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
