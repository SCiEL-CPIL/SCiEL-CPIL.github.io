"""Block Strassen decomposition for 6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = 6, 6, 6
    
    # Strassen's 7-term algorithm for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 6x6 matrices, split into 3x3 blocks of 2x2
    # A[i*2 + a, j*2 + b] where i,j in {0,1,2} are block indices, a,b in {0,1} are within-block
    
    # Each term in Strassen involves products of 2x2 sub-blocks
    # A block product (2x2 * 2x2) needs 8 scalar multiplications (schoolbook)
    # So each Strassen term (which is a sum of 2x2 blocks) * (2x2 block) needs 8 multiplications
    # 7 terms * 8 = 56 multiplications per 2x2 block pair
    # 3*3 = 9 block pairs
    # Total: 9 * 56 = 504? No, that's not right.
    
    # Let me reconsider. The Strassen decomposition operates on the block level.
    # We have a 3x3 grid of 2x2 blocks. For each pair of positions (i,j) in the 3x3 grid,
    # we need to compute C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
    
    # Actually, for 6x6 with 2x2 blocks, we have 3x3 blocks.
    # The block matrix multiplication is: C_block[i,j] = sum_{k=0}^{2} A_block[i,k] * B_block[k,j]
    # Each such product is a 2x2 * 2x2 = 2x2 multiplication, which costs 8 (schoolbook) or 7 (Strassen).
    
    # But Strassen applies recursively or at the top level. Here's the key insight:
    # We can apply Strassen's algorithm to the 3x3 block matrix multiplication by treating
    # each 2x2 block as a "scalar" and using 7 "multiplications" at the block level.
    # But each "block multiplication" is still a 2x2*2x2 = 8 scalar multiplications.
    
    # Wait, that gives 7 * 8 = 56 per block position, times 9 positions = 504. That's worse than schoolbook (216).
    
    # The correct approach: Apply Strassen to the whole 6x6 by viewing it as having structure.
    # The known result: block Strassen with 2x2 sub-blocks gives rank 189 for 6x6.
    # This is because 3^3 * 7 * 3 = 189? No: 7 * 3^3 = 189? 7*27=189. Yes!
    
    # The idea: use Strassen's 7-term formula where each "multiplication" is a 3x3 scalar-like operation,
    # but since our blocks are 2x2, each 3x3 "scalar" multiplication... 
    
    # Let me think differently. The standard block Strassen for (2^k)^2 matrices:
    # For 6x6, we can't directly use 2x2 Strassen recursively since 6 is not a power of 2.
    # But we can pad or use a hybrid approach.
    
    # Actually, the measured improvement was 216 -> 189. Let's figure out how.
    # 189 = 7 * 27 = 7 * 3^3
    # This suggests: decompose the 6x6 into 3x3 blocks of 2x2, then apply Strassen
    # at the block level (treating 2x2 blocks as atoms that multiply via 8 schoolbook terms),
    # BUT that would give 7 * 8 * 9 = 504, not 189.
    
    # Alternative: 189 = 216 * 7/8. So we replace each 2x2*2x2 multiplication (8 terms) 
    # with Strassen (7 terms), and we have 216/8 = 27 such 2x2*2x2 multiplications in schoolbook.
    # 27 * 7 = 189. Yes!
    
    # In schoolbook for 6x6: C[i,j] = sum_k A[i,k]*B[k,j], so 6*6*6 = 216 scalar mults.
    # Group into 2x2 blocks: there are 3*3*3 = 27 groups of (2x2)*(2x2), each needing 8 mults.
    # Replace each 8-mult 2x2*2x2 with Strassen's 7-mult version: 27 * 7 = 189.
    
    # So the decomposition: for each block position (I, K, J) in the 3x3 grid,
    # use 7 Strassen terms instead of 8 schoolbook terms for the 2x2*2x2 product.
    
    # Strassen terms for 2x2 blocks at position (I,K) for A and (K,J) for B:
    # Let A[I,K] = [[A00, A01],[A10, A11]], B[K,J] = [[B00, B01],[B10, B11]]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C[I,J] = [[C00, C01],[C10, C11]]
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each M_t is a scalar product of linear combinations, so in the tensor decomposition:
    # Each M_t contributes 1 rank term (it's a single "multiplication" of two linear forms)
    
    # So for each (I,K,J) block triple, we have 7 rank terms.
    # Total rank: 27 * 7 = 189.
    
    # Now construct U, V, W matrices.
    # U shape: (n*m, rank) = (36, 189)
    # V shape: (m*p, rank) = (36, 189)  
    # W shape: (p*n, rank) = (36, 189)
    
    # Indexing:
    # U[i*m+j, col] = U[i*6+j, col]
    # V[j*p+k, col] = V[j*6+k, col]
    # W[k*n+i, col] = W[k*6+i, col]
    
    # For Strassen term t at block (I,K,J):
    # The term is: (linear combo of A entries in block (I,K)) * (linear combo of B entries in block (K,J))
    # This contributes to C entries in block (I,J) via the output coefficients.
    
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen term definitions: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: 4 values for A00, A01, A10, A11
    # b_coeffs: 4 values for B00, B01, B10, B11
    # c_coeffs: 4 values for C00, C01, C10, C11 (how this term contributes to output)
    
    strassen_terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2 = (A10+A11)*B00 -> C10+=1, C11-=1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        # M3 = A00*(B10-B11) -> C01+=1, C11+=1
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        # M4 = A11*(B01-B10) -> C00+=1, C10+=1
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        # M5 = (A00+A01)*B11 -> C00-=1, C01+=1
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6 = (A10-A00)*(B00+B01) -> C11+=1
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7 = (A01-A11)*(B10+B11) -> C00+=1
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]
    
    col = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                for t in range(7):
                    a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
                    
                    # A block (I,K): rows 2I, 2I+1; cols 2K, 2K+1
                    # A00 at (2I, 2K), A01 at (2I, 2K+1), A10 at (2I+1, 2K), A11 at (2I+1, 2K+1)
                    a_positions = [
                        (2*I + 0, 2*K + 0),  # A00
                        (2*I + 0, 2*K + 1),  # A01
                        (2*I + 1, 2*K + 0),  # A10
                        (2*I + 1, 2*K + 1),  # A11
                    ]
                    for idx, (a_val, (ai, aj)) in enumerate(zip(a_coeffs, a_positions)):
                        if a_val != 0:
                            U[ai * 6 + aj][col] = a_val
                    
                    # B block (K,J): rows 2K, 2K+1; cols 2J, 2J+1
                    # B00 at (2K, 2J), B01 at (2K, 2J+1), B10 at (2K+1, 2J), B11 at (2K+1, 2J+1)
                    b_positions = [
                        (2*K + 0, 2*J + 0),  # B00
                        (2*K + 0, 2*J + 1),  # B01
                        (2*K + 1, 2*J + 0),  # B10
                        (2*K + 1, 2*J + 1),  # B11
                    ]
                    for idx, (b_val, (bi, bj)) in enumerate(zip(b_coeffs, b_positions)):
                        if b_val != 0:
                            V[bi * 6 + bj][col] = b_val
                    
                    # C block (I,J): the term contributes to C entries
                    # C00 at (2I, 2J), C01 at (2I, 2J+1), C10 at (2I+1, 2J), C11 at (2I+1, 2J+1)
                    # W[k*n+i, col] where k is the "K" index in the output, i is the "I" index
                    # Wait, W[k*n+i, col] means for output C[i,j], the contribution involves
                    # W[k*6+i, col] for each k, summed over k.
                    # But in our construction, each term is associated with a specific (I,K,J) triple.
                    # The output C[i,j] for i in block I, j in block J gets contribution from W[k*6+i, col]
                    # where k should be... hmm.
                    
                    # Actually, the tensor is T[i,j,k] = A[i,k]*B[k,j]. 
                    # The decomposition is T[i,j,k] = sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
                    # Wait, that's not standard. Let me re-read.
                    
                    # The standard: C[i,j] = sum_k A[i,k]*B[k,j]
                    # Tensor form: the multiplication tensor has entries T_{i,j,k} = 1 if the output C[i,j]
                    # depends on A[i,k]*B[k,j]. So T_{i,j,k} = delta (always 1 in the sense that 
                    # C[i,j] = sum_k A[i,k] B[k,j]).
                    
                    # The rank-r decomposition: T_{i,j,k} = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Wait, the task says: W[k*n+i, r], so W index is (k*n+i) = (k*6+i).
                    # And the formula is: sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    
                    # Hmm, that doesn't look right dimensionally. Let me re-examine.
                    # U: (n*m, rank) = (36, rank), indexed by (i,j) -> i*6+j
                    # V: (m*p, rank) = (36, rank), indexed by (j,k) -> j*6+k  
                    # W: (p*n, rank) = (36, rank), indexed by (k,i) -> k*6+i
                    
                    # And C[i,j] = sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    
                    # This is correct! For each output (i,j), we sum over k and r.
                    
                    # So for our Strassen term at block (I,K,J):
                    # The A part: linear combo of A[2I+a, 2K+b] for a,b in {0,1}
                    # The B part: linear combo of B[2K+c, 2J+d] for c,d in {0,1}
                    # The output: contributes to C[2I+e, 2J+f] for e,f in {0,1}
                    
                    # In the tensor decomposition, this term (rank col) has:
                    # U[i*6+j, col] nonzero when (i,j) is in A block (I,K)
                    # V[j*6+k, col] nonzero when (j,k) is in B block (K,J)
                    # W[k*6+i, col] nonzero when (k,i) corresponds to output C[i,j] being in block (I,J)
                    # and k is in block K.
                    
                    # Specifically, W[k*6+i, col]: for this to be nonzero, we need
                    # i in {2I, 2I+1} (output row block I) and k in {2K, 2K+1} (sum index block K)
                    # and the coefficient depends on the Strassen output coefficients.
                    
                    # The output coefficient c_coeffs[idx] applies to C[2I+e, 2J+f] where idx = e*2+f
                    # But W is indexed by (k, i) -> k*6+i. For a given (k, i), the column j of C
                    # is determined by... wait, W[k*6+i, col] is just a factor. The actual output
                    # C[i,j] gets the contribution through the product U*V*W summed over k.
                    
                    # For our term: U is nonzero for (i,j) in A-block(I,K), V is nonzero for (j,k) in B-block(K,J).
                    # So for the product U[i,j,col]*V[j,k,col]*W[k,i,col] to be nonzero, we need
                    # i in A-block(I,K) rows, j in A-block(I,K) cols AND j in B-block(K,J) rows,
                    # k in B-block(K,J) cols AND k in W's k-index.
                    
                    # j must be in {2K, 2K+1} (from A block col and B block row overlap)
                    # i must be in {2I, 2I+1} (from A block row and W's i-index)
                    # k must be in {2K, 2K+1} (from B block col and W's k-index)
                    
                    # So W[k*6+i, col] is nonzero for i in {2I,2I+1}, k in {2K,2K+1}
                    # And the value should encode which output C[i,j] this contributes to.
                    
                    # But C[i,j] has j in {2J, 2J+1}. How does j appear? It's in U and V already.
                    # The W factor W[k*6+i, col] doesn't depend on j. So how do we select the right j?
                    
                    # Ah, I see the issue. The decomposition is:
                    # C[i,j] = sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    
                    # For a fixed term r=col, the contribution to C[i,j] is:
                    # sum_k U[i*6+j, col] * V[j*6+k, col] * W[k*6+i, col]
                    
                    # = U[i*6+j, col] * sum_k V[j*6+k, col] * W[k*6+i, col]
                    
                    # For our Strassen term: U[i*6+j, col] is a linear combo coefficient for A[i,j]
                    # V[j*6+k, col] is a linear combo coefficient for B[j,k]
                    # W[k*6+i, col] is a coefficient that must make the product work out.
                    
                    # The product U[i*6+j, col] * V[j*6+k, col] * W[k*6+i, col] summed over k
                    # should give: (a_coeff * A[i,j]) * (b_coeff * B[j,k]) * (c_coeff) for the right (i,j,k)
                    
                    # But A[i,j] and B[j,k] are variables, so the coefficient must be:
                    # U[i*6+j, col] * V[j*6+k, col] * W[k*6+i, col] summed over k
                    # = a_val * b_val * c_val for the appropriate indices.
                    
                    # Since a_val, b_val, c_val are constants, and i,j,k vary, we need:
                    # For each valid (i,j,k): U[i*6+j,col]*V[j*6+k,col]*W[k*6+i,col] = a_val*b_val*c_val
                    # For invalid (i,j,k): the product should be 0.
                    
                    # This is satisfied if:
                    # U[i*6+j,col] = a_val (for (i,j) in A block)
                    # V[j*6+k,col] = b_val (for (j,k) in B block)  
                    # W[k*6+i,col] = c_val (for (k,i) with i in output row block, k in sum block)
                    
                    # But c_val depends on which output C[i,j] this is. The output index j is in block J.
                    # However, W[k*6+i, col] doesn't have j in its index!
                    
                    # This means: for a given (k,i), the same W value is used for all j.
                    # So the output coefficient must be independent of j, or we need multiple terms.
                    
                    # Looking at Strassen: each M_t contributes to specific (C00, C01, C10, C11) entries
                    # with specific coefficients. The (e,f) index determines which output.
                    # i = 2I+e, j = 2J+f.
                    
                    # W[k*6+i, col] only depends on i (and k), not j. So for fixed i, all j get the same W factor.
                    # This means we can't distinguish C[i,2J] from C[i,2J+1] using a single W value.
                    
                    # Solution: we need separate rank terms for each output column within the block,
                    # OR the Strassen structure naturally handles this.
                    
                    # Actually, re-examining: the A part selects (i,j) in block (I,K), so j is in block K.
                    # The B part selects (j,k) in block (K,J), so k is in block J.
                    # The output C[i,j] has i in block I (row) and j in block K (col from A).
                    # Wait, no! C[i,j] where i is row index, j is col index. 
                    # A is (i,k), B is (k,j). So A's col index = B's row index = k (sum index).
                    # C's row = i, C's col = j.
                    
                    # In our block notation: I is block row of A/C, K is block col of A / block row of B,
                    # J is block col of B/C.
                    
                    # So A block (I,K): A[2I+a, 2K+b]
                    # B block (K,J): B[2K+c, 2J+d]
                    # C block (I,J): C[2I+e, 2J+f]
                    
                    # The product A[i,k]*B[k,j] with i in block I, k in block K, j in block J.
                    # U[i*6+k, col] for A entry (i,k)
                    # V[k*6+j, col] for B entry (k,j)
                    # W[j*6+i, col] for... wait, W[k*n+i, col] = W[k*6+i, col]
                    
                    # Hmm, W is indexed by (k, i) -> k*6+i. But k is the sum index, i is the output row.
                    # For the term to contribute to C[i,j], we need the product summed over k:
                    # sum_k U[i*6+k, col] * V[k*6+j, col] * W[k*6+i, col]
                    
                    # Wait, I had it wrong. Let me re-read the task:
                    # U[i*m+j, col], V[j*p+k, col], W[k*n+i, col]
                    # So U is indexed by (i,j) -> i*6+j, where i is A's row, j is A's col
                    # V is indexed by (j,k) -> j*6+k, where j is B's row, k is B's col
                    # W is indexed by (k,i) -> k*6+i, where k is... ?, i is... ?
                    
                    # And the formula: C[i,j] = sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    
                    # So for output C[i,j]: sum over k and r.
                    # U[i*6+j, r]: A's (i,j) entry
                    # V[j*6+k, r]: B's (j,k) entry
                    # W[k*6+i, r]: some factor indexed by (k,i)
                    
                    # The product U[i,j,r]*V[j,k,r]*W[k,i,r] summed over k gives C[i,j] contribution.
                    
                    # For our Strassen term: 
                    # A entry (i,k) where i in block I rows, k in block K cols: U[i*6+k, col]
                    # B entry (k,j) where k in block K rows, j in block J cols: V[k*6+j, col]
                    # W[k*6+i, col]: indexed by (k,i) where k in block K, i in block I
                    
                    # The output C[i,j] has i in block I, j in block J.
                    # W[k*6+i, col] doesn't depend on j, so for fixed (k,i), all j in block J get same W factor.
                    # This is problematic unless the output coefficients are the same for all j in the block.
                    
                    # In Strassen, the output goes to C[I,J] block, which is 2x2. The coefficients
                    # c_coeffs[0..3] apply to C00, C01, C10, C11 of that block, i.e., C[2I,2J], C[2I,2J+1], etc.
                    
                    # Since W doesn't index j, we can't distinguish j=2J from j=2J+1 in W.
                    # But V[k*6+j, col] does depend on j! And U[i*6+k, col] doesn't depend on j either
                    # (it depends on i and k).
                    
                    # Wait, U[i*6+j, col] where j is A's column. In our case, A's column is k (the sum index).
                    # I think I'm confusing myself. Let me restart the index mapping.
                    
                    # C[i,j] = sum_k A[i,k] * B[k,j]
                    # U[i*m + j, r] where (i,j) = (row of A, col of A) = (i, k) -- NO, j is col of A which is k in C[i,j]=sum_k...
                    
                    # I think the notation in the task uses:
                    # U indexed by (i, j) corresponding to A[i,j]
                    # V indexed by (j, k) corresponding to B[j,k]  
                    # W indexed by (k, i)
                    # And C[i,k] = sum_j U[i,j,r]*V[j,k,r]*W[k,i,r]? No...
                    
                    # Let me just use: the formula is C[i,j] = sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    # This means: for output (i,j), sum over intermediate k.
                    # U[i*6+j, r]: factor for A's (i, ?) -- here j appears, so maybe A's col is j?
                    # But then V[j*6+k, r]: B's (j, k)
                    # W[k*6+i, r]: factor for (k, i)
                    
                    # If A is 6x6 with A[row, col], and C = A*B, then C[i,j] = sum_k A[i,k]*B[k,j].
                    # Comparing: U[i*6+j, r] should correspond to A[i, ?]. If j is A's col, then A's col = j,
                    # but in C[i,j]=sum_k A[i,k]*B[k,j], A's col is k, not j.
                    
                    # I think the task's indexing is:
                    # U[i*m+j, r] where this is associated with A[i,j] (i=row, j=col of A)
                    # V[j*p+k, r] associated with B[j,k] (j=row, k=col of B)
                    # W[k*n+i, r] associated with... the output?
                    # And C[i,k] = sum_j U[i,j,r]*V[j,k,r]*W[k,i,r]?
                    
                    # Actually the task says "the output factor follows the official trace convention W[k*n+i,r]"
                    # and the formula should be C[i,j] = sum_k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                    # but that has i appearing in both U and W with different meanings, which is suspicious.
                    
                    # Let me just go with: C[i,j] = sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    # and verify with schoolbook.
                    
                    # In schoolbook: for each (i,j,k), one term with U[i*6+k, col]=1, V[k*6+j, col]=1, W[j*6+i, col]=1
                    # Wait, the current program has:
                    # U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1 for each (i,j,k)
                    # So U[i*6+j, col]=1 where j is A's col index
                    # V[j*6+k, col]=1 where j is B's row index, k is B's col index
                    # W[k*6+i, col]=1 where k is... , i is...
                    
                    # In the loop: for i in range(n), for j in range(m), for k in range(p):
                    # U[i*m+j, col] = U[i*6+j, col] -- i is A row, j is A col
                    # V[j*p+k, col] = V[j*6+k, col] -- j is B row, k is B col
                    # W[k*n+i, col] = W[k*6+i, col] -- k is B col, i is A row
                    
                    # So the formula is: C[i, k] = sum_j sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
                    # Where i is output row, k is output col, j is sum index.
                    
                    # This makes sense! C[i,k] = sum_j A[i,j]*B[j,k], with U for A[i,j], V for B[j,k], W for (k,i).
                    
                    # So in our Strassen construction:
                    # A block (I,K): A[2I+a, 2K+b], so i=2I+a (A row), j=2K+b (A col)
                    # B block (K,J): B[2K+c, 2J+d], so j=2K+c (B row), k=2J+d (B col)
                    # Note: A's col j must equal B's row j, so both are in block K.
                    # Output C[i,k] where i=2I+e (row), k=2J+f (col)
                    # W[k*6+i, col] = W[(2J+f)*6+(2I+e), col]
                    
                    # Now W depends on (k,i) = (2J+f, 2I+e). The output coefficients c_coeffs
                    # apply to (e,f) pairs. Since W's index includes both e (via i) and f (via k),
                    # we CAN set W differently for each (e,f). 
                    
                    # So for Strassen term t at block (I,K,J):
                    # For each (a,b) in A block: U[(2I+a)*6+(2K+b), col] = a_coeffs[idx]
                    # For each (c,d) in B block: V[(2K+c)*6+(2J+d), col] = b_coeffs[idx]
                    # For each (e,f) in C block: W[(2J+f)*6+(2I+e), col] = c_coeffs[idx]
                    # where idx_a = a*2+b, idx_b = c*2+d, idx_c = e*2+f
                    
                    # Let me redo the construction.
                    
    # Clear and rebuild
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    col = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                for t in range(7):
                    a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
                    for a in range(2):
                        for b in range(2):
                            a_val = a_coeffs[a*2+b]
                            if a_val != 0:
                                i = 2*I + a
                                j = 2*K + b
                                U[i*6+j][col] = a_val
                    for c in range(2):
                        for d in range(2):
                            b_val = b_coeffs[c*2+d]
                            if b_val != 0:
                                j2 = 2*K + c
                                k = 2*J + d
                                V[j2*6+k][col] = b_val
                    for e in range(2):
                        for f in range(2):
                            c_val = c_coeffs[e*2+f]
                            if c_val != 0:
                                i_out = 2*I + e
                                k_out = 2*J + f
                                W[k_out*6+i_out][col] = c_val
                    col += 1
    
    # Verify
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_recon = np.zeros((6,6))
    for i in range(6):
        for k in range(6):
            s = 0.0
            for j in range(6):
                for r in range(rank):
                    s += U_arr[i*6+j, r] * V_arr[j*6+k, r] * W_arr[k*6+i, r]
            C_recon[i,k] = s
    
    # The multiplication tensor T[i,k,j] = 1 (A[i,j]*B[j,k])
    # C_recon[i,k] should equal sum_j T[i,k,j] = 6? No, T is the tensor, not a value.
    # The decomposition should satisfy: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # (since T[i,j,k] = 1 for the multiplication tensor)
    
    max_err = 0.0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*6+j, r] * V_arr[j*6+k, r] * W_arr[k*6+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-9:
        # Fall back to schoolbook
        rank = 216
        U = [[0]*rank for _ in range(36)]
        V = [[0]*rank for _ in range(36)]
        W = [[0]*rank for _ in range(36)]
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i*6+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*6+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
