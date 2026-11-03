"""Matrix multiplication 6x6x6: Strassen-based rank-216 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a rank-216 exact tensor decomposition for 6x6 x 6x6 matrix multiplication.
    
    Uses a recursive Strassen-style approach:
    - 6x6 = 3x3 blocks of size 2x2
    - Each 2x2 uses Strassen (7 multiplications)
    - Total: 3*3*7 = 63 scalar multiplications for the "inner" part
    - But we need to account for the block structure properly.
    
    Actually, let me think more carefully.
    
    For 6x6 matrix multiplication, we can view matrices as 3x3 blocks of 2x2 matrices.
    If we apply Strassen at the 2x2 level within each block position, we get:
    - The 6x6 multiplication tensor has rank at most 3^3 * 7 = 189 if we use Strassen on the 3x3 block structure... no wait.
    
    Let me reconsider. The standard approach:
    - 6x6 matrices can be seen as 2x2 blocks of 3x3 matrices, or 3x3 blocks of 2x2 matrices.
    
    If we use 3x3 blocks of 2x2:
    - We have 9 block positions, each a 2x2 matrix multiplication
    - Each 2x2 multiplication can be done with 7 scalar multiplications (Strassen)
    - But the blocks interact: C[i,j] = sum_k A[i,k] * B[k,j] where A[i,k], B[k,j], C[i,j] are 2x2 blocks
    - So C_block[i,j] = sum over k of A_block[i,k] * B_block[k,j]
    - This is a 3x3 "matrix multiplication" of 2x2 matrices
    - The rank of 3x3 matrix multiplication is at most 3^3 = 27 (schoolbook) or can be reduced
    
    Actually, the tensor for 3x3 matrix multiplication has rank at most 27 (schoolbook).
    If we use Strassen recursively: 3x3 doesn't directly apply Strassen (which is for 2x2).
    
    Alternative: View 6x6 as 2x2 blocks of 3x3.
    - 2x2 block multiplication uses Strassen: 7 multiplications of 3x3 matrices
    - Each 3x3 multiplication has rank at most 27 (schoolbook)
    - Total rank: 7 * 27 = 189
    
    Or view as 3x3 blocks of 2x2:
    - 3x3 block multiplication: schoolbook rank 27, each involving 2x2 multiplication
    - Each 2x2 multiplication has rank 7 (Strassen)
    - Total rank: 27 * 7 = 189
    
    Either way, rank 189 is achievable.
    
    Let me implement the 2x2-block-of-3x3 approach with Strassen:
    
    A = [[A11, A12], [A21, A22]] where each Ai is 3x3
    B = [[B11, B12], [B21, B22]]
    
    Strassen:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 3x3 matrix product, done with schoolbook (27 scalar mults).
    Total: 7 * 27 = 189 scalar multiplications.
    
    Now I need to express this as U, V, W matrices of shape (36, 189), (36, 189), (36, 189).
    
    Indexing: A is 6x6, so A[i,j] for i,j in 0..5.
    Block A11 = A[0:3, 0:3], A12 = A[0:3, 3:6], A21 = A[3:6, 0:3], A22 = A[3:6, 3:6]
    
    For each of the 7 Strassen multiplications, and for each of the 27 scalar multiplications within a 3x3 product, we get one term.
    
    Let me index:
    - Strassen factor s in 0..6
    - Within each 3x3 product, indices (ii, jj, kk) in 0..2 for row, col, sum-index
    
    For factor s, the product is (A_s) * (B_s) where A_s and B_s are linear combinations of blocks.
    
    Let me define the linear combinations for each Strassen factor:
    
    s=0: M1 = (A11+A22)*(B11+B22)
      A_s: blocks with coeff: A11:1, A22:1, others:0
      B_s: blocks with coeff: B11:1, B22:1, others:0
    
    s=1: M2 = (A21+A22)*B11
      A_s: A21:1, A22:1
      B_s: B11:1
    
    s=2: M3 = A11*(B12-B22)
      A_s: A11:1
      B_s: B12:1, B22:-1
    
    s=3: M4 = A22*(B21-B11)
      A_s: A22:1
      B_s: B21:1, B11:-1
    
    s=4: M5 = (A11+A12)*B22
      A_s: A11:1, A12:1
      B_s: B22:1
    
    s=5: M6 = (A21-A11)*(B11+B12)
      A_s: A21:1, A11:-1
      B_s: B11:1, B12:1
    
    s=6: M7 = (A12-A22)*(B21+B22)
      A_s: A12:1, A22:-1
      B_s: B21:1, B22:1
    
    For each factor s, and for each (ii, jj, kk) in 0..2:
      The scalar multiplication is: sum over the contributing elements.
      
      A_s[ii,jj] = sum over blocks (bi, bj) of coeff_A_s(bi,bj) * A[3*bi+ii, 3*bj+jj]
      B_s[kk, ll] = sum over blocks (bk, bl) of coeff_B_s(bk,bl) * B[3*bk+kk, 3*bl+ll]
      
      Wait, I need to be more careful with indices.
      
      A is 6x6. Block A(bi,bj) where bi,bj in {0,1} is A[3*bi:3*bi+3, 3*bj:3*bj+3].
      Element A[3*bi+ii, 3*bj+jj] for ii,jj in 0..2.
      
      Similarly for B.
      
      The product M_s[ii, ll] = sum_kk A_s[ii, kk] * B_s[kk, ll]
      
      Where A_s[ii, kk] = sum_{bi,bj} cA_s(bi,bj) * A[3*bi+ii, 3*bj+kk]
      And B_s[kk, ll] = sum_{bk,bl} cB_s(bk,bl) * B[3*bk+kk, 3*bl+ll]
      
      So M_s[ii, ll] = sum_kk sum_{bi,bj} sum_{bk,bl} cA_s(bi,bj) * cB_s(bk,bl) * A[3*bi+ii, 3*bj+kk] * B[3*bk+kk, 3*bl+ll]
      
      And C[3*ci+ii, 3*cj+ll] = sum_s coeff_C(s, ci, cj) * M_s[ii, ll]
      
      Where coeff_C(s, ci, cj) is the coefficient of M_s in C_block(ci,cj).
      
    For the tensor decomposition, each scalar multiplication corresponds to one column of U, V, W.
    
    The scalar mult for factor s, indices (ii, kk, ll) is:
      A_part = A[3*bi+ii, 3*bj+kk] for each (bi,bj) with cA_s(bi,bj) != 0
      B_part = B[3*bk+kk, 3*bl+ll] for each (bk,bl) with cB_s(bk,bl) != 0
      
      Actually, each "scalar multiplication" in the schoolbook 3x3 product is:
      A_s[ii, kk] * B_s[kk, ll]
      
      But A_s[ii, kk] is itself a linear combination of A elements, and B_s[kk, ll] is a linear combination of B elements.
      
      For the tensor decomposition, we need to expand each such product into elementary terms.
      
      A_s[ii, kk] * B_s[kk, ll] = (sum_{bi,bj} cA_s(bi,bj) * A[3*bi+ii, 3*bj+kk]) * (sum_{bk,bl} cB_s(bk,bl) * B[3*bk+kk, 3*bl+ll])
      
      This is a product of two linear forms, which in the tensor decomposition becomes multiple terms (one for each pair of elementary factors).
      
      Hmm, this means the rank is not just 7*27=189. Let me reconsider.
      
      Actually, in the tensor decomposition framework, each "rank-1 term" is u_i * v_j * w_k where u is a vector in A-space, v in B-space, w in C-space.
      
      When we write A_s[ii,kk] * B_s[kk,ll], if A_s[ii,kk] = sum_a alpha_a * A[a] and B_s[kk,ll] = sum_b beta_b * B[b], then the product is sum_a sum_b alpha_a * beta_b * A[a] * B[b].
      
      Each term A[a]*B[b] contributes to C via the coefficient structure. So each such (a,b) pair is a rank-1 term in the decomposition.
      
      So the total rank is sum over s, ii, kk, ll of (number of nonzero A_s coefficients) * (number of nonzero B_s coefficients).
      
      Let me count:
      s=0: A_s has 2 nonzero blocks (each with 9 elements, but for fixed ii,kk, it's 2 terms), B_s has 2 nonzero blocks (2 terms for fixed kk,ll). So 2*2=4 terms per (ii,kk,ll).
      
      Wait, for fixed ii and kk, A_s[ii,kk] = cA(bi=0,bj=0)*A[ii,kk] + cA(bi=1,bj=1)*A[3+ii,3+kk] (for s=0). So 2 terms.
      Similarly B_s[kk,ll] has 2 terms.
      So 2*2=4 rank-1 terms per (ii,kk,ll) for s=0.
      
      s=1: A_s has 2 terms, B_s has 1 term. 2*1=2 per (ii,kk,ll).
      s=2: A_s has 1 term, B_s has 2 terms. 1*2=2 per (ii,kk,ll).
      s=3: A_s has 1 term, B_s has 2 terms. 1*2=2 per (ii,kk,ll).
      s=4: A_s has 2 terms, B_s has 1 term. 2*1=2 per (ii,kk,ll).
      s=5: A_s has 2 terms, B_s has 2 terms. 2*2=4 per (ii,kk,ll).
      s=6: A_s has 2 terms, B_s has 2 terms. 2*2=4 per (ii,kk,ll).
      
      Total per (ii,kk,ll): 4+2+2+2+2+4+4 = 20 terms.
      Number of (ii,kk,ll) triples: 3*3*3 = 27.
      Total rank: 20*27 = 540.
      
      That's worse than schoolbook (216)! I made an error in my reasoning.
      
      The issue is that when we "multiply" two linear combinations, we get a bilinear form that requires the product of the number of terms in each.
      
      Let me reconsider the approach. Maybe I should use a different block size or a different method.
      
      Alternative: Use 2x2 blocks of 3x3, but do the 3x3 multiplication with schoolbook, and handle the Strassen linear combinations separately.
      
      Actually, the correct way to count: The tensor for matrix multiplication of size n has rank at most n^3 (schoolbook). Strassen shows rank at most 2^n * 7^(log2(n))... no, Strassen shows that n x n multiplication can be done with O(n^2.807) operations, but the tensor rank is different.
      
      For 6x6, the best known upper bound on tensor rank is... let me think. 6 = 2*3. 
      
      Actually, I recall that the rank of 6x6 matrix multiplication tensor is at most 216 (schoolbook) and can be improved. Let me try a different decomposition.
      
      Let me just use the schoolbook decomposition but see if I can find a better one. The schoolbook rank is 216.
      
      Wait, maybe I should try using the fact that 6 = 2*3 and use a "mixed" approach.
      
      Actually, let me reconsider. The Strassen algorithm for 2x2 uses 7 multiplications. If I tile 6x6 into 2x2 blocks of 3x3, then:
      - I need to compute C = A*B where A, B, C are 2x2 block matrices with 3x3 blocks.
      - Using Strassen, I need 7 products of 3x3 matrices.
      - Each 3x3 product has tensor rank 27 (schoolbook).
      - But the "inputs" to each 3x3 product are linear combinations of the original blocks.
      
      The key insight: In the tensor decomposition, each rank-1 term is a product of a single A element, a single B element, and a single C element (with coefficients). So when I say "7 products of 3x3 matrices", each 3x3 product is itself a sum of 27 rank-1 terms, but the inputs are linear combinations.
      
      For a 3x3 product where the input matrices are linear combinations of the original 3x3 blocks, each of the 27 scalar multiplications involves products of linear combinations, which expand to multiple rank-1 terms.
      
      So my count of 540 might be correct for this particular decomposition, which is worse than schoolbook.
      
      Let me try a different approach: tile into 3x3 blocks of 2x2.
      - A, B, C are 3x3 block matrices with 2x2 blocks.
      - C[i,j] = sum_k A[i,k] * B[k,j] where each is a 2x2 product.
      - This is a 3x3 "matrix multiplication" at the block level.
      - The block-level multiplication has rank 27 (schoolbook for 3x3).
      - Each 2x2 product has rank 7 (Strassen).
      - But again, the inputs to each 2x2 product are individual blocks (not linear combinations), so each 2x2 product is exactly 7 rank-1 terms.
      - Total: 27 * 7 = 189.
      
      This is better! Let me verify:
      - Block A[i,k] is a 2x2 matrix, with elements A[2*i:2*i+2, 2*k:2*k+2].
      - Block B[k,j] is a 2x2 matrix.
      - Block C[i,j] = sum_k A[i,k] * B[k,j].
      - For each (i,k,j) triple (27 triples), we compute A[i,k] * B[k,j] using Strassen (7 multiplications).
      - Each Strassen multiplication is a product of two linear combinations of the 4 elements of the 2x2 blocks.
      
      Wait, same issue. Strassen for 2x2:
      M1 = (A11+A22)(B11+B22)
      etc.
      
      Each M is a product of two linear combinations of 2x2 elements. For example, (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22. That's 4 rank-1 terms, not 1.
      
      So the rank is not 27*7=189. It's 27 * (sum over 7 Strassen factors of (num_A_terms * num_B_terms)).
      
      For Strassen on 2x2:
      M1: 2*2=4
      M2: 2*1=2
      M3: 1*2=2
      M4: 1*2=2
      M5: 2*1=2
      M6: 2*2=4
      M7: 2*2=4
      Total: 4+2+2+2+2+4+4 = 20 per 2x2 product.
      
      So total rank: 27 * 20 = 540. Same as before!
      
      I see the issue. The "7 multiplications" in Strassen are not rank-1 terms in the tensor decomposition sense; they are bilinear forms that each expand to multiple rank-1 terms.
      
      So Strassen doesn't directly give a lower tensor rank in this naive expansion. The tensor rank of 2x2 matrix multiplication is actually 7, but that's because the 7 Strassen multiplications can be arranged so that each one is a "pure" multiplication of linear forms, and the linear forms don't overlap in a way that creates extra terms... 
      
      Wait, I think I'm confusing myself. Let me look at this more carefully.
      
      The tensor rank of 2x2 matrix multiplication is 7. This means there exist 7 rank-1 tensors whose sum is the 2x2 matrix multiplication tensor. Each rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors.
      
      In Strassen's algorithm, the 7 "multiplications" are:
      M1 = (A11+A22)(B11+B22)
      etc.
      
      Each Mi is a scalar (in the context of 2x2 with scalar entries). The "multiplication" (A11+A22)(B11+B22) is the product of two linear forms in A and two linear forms in B. In the tensor decomposition, this corresponds to a single rank-1 term where:
      - The A-part is the linear form A11+A22
      - The B-part is the linear form B11+B22
      - The C-part is the contribution to the output
      
      So each Strassen multiplication IS a single rank-1 term in the tensor decomposition! The linear forms are part of the u, v, w vectors.
      
      I was wrong to expand the products. In the tensor decomposition, u can be a linear combination of basis vectors. So (A11+A22) is a valid u vector, and (B11+B22) is a valid v vector.
      
      So the rank of 2x2 matrix multiplication is indeed 7, with each Strassen multiplication being one rank-1 term.
      
      Similarly, for the 3x3 block tiling:
      - 27 block-level multiplications (i,k,j triples)
      - Each block-level multiplication is a 2x2 matrix multiplication with rank 7
      - Total rank: 27 * 7 = 189
      
      This should work! Let me implement it.
      
      For each (i, k, j) in 0..2:
        For each Strassen factor s in 0..6:
          One rank-1 term.
          
          U part: linear combination of A[2*i+ii, 2*k+jj] for the appropriate (ii, jj) based on Strassen factor s.
          V part: linear combination of B[2*k+kk, 2*j+ll] for the appropriate (kk, ll) based on Strassen factor s.
          W part: contribution to C[2*i+cc, 2*j+dd] based on Strassen factor s.
      
      Let me define the Strassen factors for 2x2:
      A blocks: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
      B blocks: B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
      C blocks: C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
      
      s=0: M1=(A11+A22)(B11+B22)
        A: A11:1, A22:1
        B: B11:1, B22:1
        C: C11:+1, C22:+1 (M1 contributes +M1 to C11 and C22)
      
      Wait, let me recheck Strassen:
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
      
      So:
      s=0 (M1): C11:+1, C22:+1
      s=1 (M2): C21:+1, C22:-1
      s=2 (M3): C12:+1, C22:+1
      s=3 (M4): C11:+1, C21:+1
      s=4 (M5): C11:-1, C12:+1
      s=5 (M6): C22:+1
      s=6 (M7): C11:+1
      
      Let me verify C11: M1+M4-M5+M7 = 1+1-1+1 = 2? That's wrong.
      
      Let me recheck Strassen's algorithm:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
      
      Let me verify with a simple example. A = [[1,0],[0,1]], B = [[1,0],[0,1]].
      A11=1, A12=0, A21=0, A22=1
      B11=1, B12=0, B21=0, B22=1
      
      M1 = (1+1)(1+1) = 4
      M2 = (0+1)*1 = 1
      M3 = 1*(0-1) = -1
      M4 = 1*(0-1) = -1
      M5 = (1+0)*1 = 1
      M6 = (0-1)*(1+0) = -1
      M7 = (0-1)*(0+1) = -1
      
      C11 = 4 + (-1) - 1 + (-1) = 1 ✓
      C12 = -1 + 1 = 0 ✓
      C21 = 1 + (-1) = 0 ✓
      C22 = 4 - 1 + (-1) + (-1) = 1 ✓
      
      Good. So the C coefficients are:
      s=0 (M1): C11:+1, C22:+1
      s=1 (M2): C21:+1, C22:-1
      s=2 (M3): C12:+1, C22:+1
      s=3 (M4): C11:+1, C21:+1
      s=4 (M5): C11:-1, C12:+1
      s=5 (M6): C22:+1
      s=6 (M7): C11:+1
      
      Now for the A and B linear forms:
      s=0: A: A11:1, A22:1; B: B11:1, B22:1
      s=1: A: A21:1, A22:1; B: B11:1
      s=2: A: A11:1; B: B12:1, B22:-1
      s=3: A: A22:1; B: B21:1, B11:-1
      s=4: A: A11:1, A12:1; B: B22:1
      s=5: A: A21:1, A11:-1; B: B11:1, B12:1
      s=6: A: A12:1, A22:-1; B: B21:1, B22:1
      
      Now, for the full 6x6 decomposition:
      - Block indices i, k, j in 0..2
      - Block A[i,k] has elements at A[2*i+ii, 2*k+jj] for ii, jj in 0,1
        - A11 -> ii=0, jj=0
        - A12 -> ii=0, jj=1
        - A21 -> ii=1, jj=0
        - A22 -> ii=1, jj=1
      - Block B[k,j] has elements at B[2*k+kk, 2*j+ll] for kk, ll in 0,1
        - B11 -> kk=0, ll=0
        - B12 -> kk=0, ll=1
        - B21 -> kk=1, ll=0
        - B22 -> kk=1, ll=1
      - Block C[i,j] has elements at C[2*i+cc, 2*j+dd] for cc, dd in 0,1
        - C11 -> cc=0, dd=0
        - C12 -> cc=0, dd=1
        - C21 -> cc=1, dd=0
        - C22 -> cc=1, dd=1
      
      For each (i, k, j, s) quadruple:
        U vector: nonzero at positions (2*i+ii, 2*k+jj) for (ii,jj) in the A linear form of factor s.
        V vector: nonzero at positions (2*k+kk, 2*j+ll) for (kk,ll) in the B linear form of factor s.
        W vector: nonzero at positions (2*i+cc, 2*j+dd) for (cc,dd) in the C linear form of factor s.
      
      Total rank: 3*3*3*7 = 189.
      
      Let me implement this.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen factors: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict from (ii, jj) to coeff
    # B_coeffs: dict from (kk, ll) to coeff
    # C_coeffs: dict from (cc, dd) to coeff
    strassen = [
        # s=0: M1=(A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s=1: M2=(A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s=2: M3=A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s=3: M4=A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s=4: M5=(A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s=5: M6=(A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s=6: M7=(A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    A_coeffs, B_coeffs, C_coeffs = strassen[s]
                    
                    # U: A[2*i+ii, 2*k+jj]
                    for (ii, jj), coeff in A_coeffs.items():
                        row = (2 * i + ii) * m + (2 * k + jj)
                        U[row][col] = coeff
                    
                    # V: B[2*k+kk, 2*j+ll]
                    for (kk, ll), coeff in B_coeffs.items():
                        row = (2 * k + kk) * p + (2 * j + ll)
                        V[row][col] = coeff
                    
                    # W: C[2*i+cc, 2*j+dd] -> index is (2*j+dd)*n + (2*i+cc) per trace convention W[k*n+i, r]
                    # Wait, the task says W[k*n+i, r] where k is the column index of C and i is the row index.
                    # C is p x n = 6x6. C[row, col] = C[2*i+cc, 2*j+dd].
                    # W index: k*n + i where k is col of C, i is row of C.
                    # So W[(2*j+dd)*n + (2*i+cc), col] = coeff
                    for (cc, dd), coeff in C_coeffs.items():
                        row = (2 * j + dd) * n + (2 * i + cc)
                        W[row][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
