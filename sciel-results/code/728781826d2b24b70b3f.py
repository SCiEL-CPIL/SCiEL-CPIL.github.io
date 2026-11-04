"""Matrix multiplication 6x6x6 via Laderman 4x4 block decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS

    # We'll build U, V, W as dictionaries mapping (row, col) -> value
    # then convert to lists at the end.
    # U: rows = n*m = 36, V: rows = m*p = 36, W: rows = p*n = 36

    U_dict = {}
    V_dict = {}
    W_dict = {}

    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        """Add a rank-1 term with given coefficients."""
        if u_val == 0:
            return
        U_dict[(u_row, 0)] = U_dict.get((u_row, 0), 0) + u_val
        V_dict[(v_row, 0)] = V_dict.get((v_row, 0), 0) + v_val
        W_dict[(w_row, 0)] = W_dict.get((w_row, 0), 0) + w_val

    # We'll use a list of rank-1 terms instead, then assemble.
    terms = []  # list of (u_row, u_val, v_row, v_val, w_row, w_val)

    def add(u_row, u_val, v_row, v_val, w_row, w_val):
        terms.append((u_row, u_val, v_row, v_val, w_row, w_val))

    # Laderman's algorithm for 4x4 matrix multiplication (rank 20)
    # Splits 4x4 into four 2x2 blocks and uses 20 products.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = A*B, C[i,j] is 2x2
    #
    # Laderman's 20 products (using block indices 1-4 for 2x2 subblocks):
    # Let's index 2x2 blocks as (row_block, col_block) with 0,1 for the two 2x2 blocks
    #
    # The 20 Laderman products for 4x4:
    # p1  = A11 * B11
    # p2  = (A21+A11) * B12
    # p3  = (A12-A22) * B22
    # p4  = A22 * (B22+B12)
    # p5  = (A11+A12) * (B21-B22)
    # p6  = (A22+A21) * (B11-B12)
    # p7  = (A12+A22) * (B21+B11)
    # p8  = (A11-A21) * (B21+B22)
    # p9  = (A12+A11) * (B22+B12)
    # p10 = (A22+A21) * (B11+B21)
    # p11 = A11 * (B11-B21)
    # p12 = (A11+A21) * B21
    # p13 = (A12+A22) * B12
    # p14 = A22 * (B22-B12)
    # p15 = (A22-A12) * (B22+B12)
    # p16 = (A11-A21) * (B21-B22)
    # p17 = (A12+A22) * (B21-B11)
    # p18 = (A11+A12) * (B21+B11)
    # p19 = (A22+A21) * (B12+B11)
    # p20 = (A22-A12) * (B12+B22)

    # Actually, let me use a well-known set of 20 Laderman products.
    # The standard Laderman 4x4 decomposition uses 20 scalar multiplications
    # when the 4x4 is viewed as a 2x2 block matrix with 2x2 entries.
    # Each "product" is a 2x2 matrix product, which itself requires 4 scalar products
    # in the schoolbook, but in the tensor decomposition we count them as rank-1 terms.

    # Wait - I need to think about this more carefully. The tensor decomposition
    # of 6x6x6 matrix multiplication has 216 rank-1 terms in schoolbook.
    # If I decompose 6x6 as a 3x3 grid of 2x2 blocks, I get 9 block positions.
    # But Laderman works on 4x4 (2x2 grid of 2x2 blocks).

    # Alternative approach: decompose 6x6 using a 3x3 tiling of 2x2 blocks.
    # The product C = A*B where A,B are 6x6.
    # If I tile A and B into 3x3 grids of 2x2 blocks, then C is also 3x3 of 2x2 blocks.
    # C[i,j] = sum_k A[i,k] * B[k,j] (block multiplication, each is 2x2*2x2)
    # Each 2x2*2x2 product has rank 4 (schoolbook) or rank 7 (Strassen).
    # Total: 9 positions * 3 inner sum * 4 = 108 rank-1 terms (schoolbook)
    # Or with Strassen: 9 * 3 * 7 = 189 - that's worse.

    # Hmm, let me reconsider. The schoolbook for 6x6 is 216 terms.
    # Using 2x2 Strassen for the inner 2x2 products: 9*3*7 = 189. Still worse than 216? No, 189 < 216.
    # Wait, 216 = 6*6*6. With Strassen 2x2: 9*3*7 = 189. Yes, improvement.

    # But can we do better with a different tiling?
    # What about tiling 6x6 as overlapping 4x4 regions and using Laderman?

    # Let me try a different approach: use the fact that 6 = 2+2+2 and 
    # apply a tensor product / Kronecker structure.

    # Actually, the simplest improvement: use Strassen for 2x2 blocks.
    # 6x6 = 3x3 grid of 2x2 blocks.
    # C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j]
    # Each A[i,k]*B[k,j] is a 2x2*2x2 product (4 scalar mults schoolbook, 7 Strassen)
    # Total with Strassen: 3*3*3*7 = 189
    # Total schoolbook: 3*3*3*4 = 108... wait that's 108 not 216.
    # 6*6*6 = 216. 3*3*3*4 = 108. That doesn't add up.
    # Oh wait, each 2x2*2x2 product has 4 scalar products, and there are 9*3=27 such products.
    # 27*4 = 108. But 6*6*6 = 216. 
    # I see my error: 3*3*3 = 27 block products, each 2x2*2x2 = 4 scalar products. 27*4=108.
    # But 6*6*6 = 216. So the schoolbook should give 216, not 108.
    # Let me recheck: A is 6x6, B is 6x6, C is 6x6.
    # C[i,j] = sum_k A[i,k]*B[k,j]. There are 6*6*6 = 216 such products.
    # If I group into 2x2 blocks: A has 9 blocks, B has 9 blocks, C has 9 blocks.
    # C_block[i,j] = sum_{k=0}^{2} A_block[i,k] * B_block[k,j]
    # Each A_block[i,k]*B_block[k,j] is a 2x2 * 2x2 matrix product = 4 scalar products.
    # Number of block products: 3*3*3 = 27.
    # Total scalar products: 27 * 4 = 108.
    # But 216 != 108. What's wrong?
    # 2x2 * 2x2 = sum over 2 intermediate indices, so 2*2*2 = 8 scalar products per block product.
    # No wait: (2x2)*(2x2) result is 2x2, each entry is sum of 2 products. So 4 entries * 2 = 8 scalar products.
    # So 27 * 8 = 216. Yes!
    # So each 2x2*2x2 block product has 8 scalar products (4 entries, each a dot product of length 2).
    # With Strassen for 2x2: 7 scalar products instead of 8.
    # Total: 27 * 7 = 189.
    # Improvement from 216 to 189.

    # Let me implement Strassen-based decomposition.
    # Strassen 2x2: 
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # In the tensor decomposition, each Strassen "product" like (A11+A22)*(B11+B22) 
    # expands to 4 rank-1 terms in the scalar tensor:
    # (a11+a22)*(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # So each Strassen multiplication of 2x2 matrices contributes 4 rank-1 terms to the overall tensor.
    # Wait, no. In the tensor decomposition, a "product" (x_i * y_j) is one rank-1 term.
    # Strassen replaces 8 such products with 7 "combined" products, but each combined product
    # like (x1+x2)*(y1+y2) is NOT a single rank-1 term - it's 4 rank-1 terms.
    # So Strassen doesn't directly reduce the rank of the tensor decomposition!
    # 
    # The rank of the tensor is the minimum number of rank-1 terms. Strassen's algorithm
    # uses 7 multiplications, but those are multiplications of linear combinations, which
    # in the tensor decomposition context means 7*4 = 28 rank-1 terms vs 8*4 = 32.
    # Hmm, that's not right either.

    # Let me think again. The matrix multiplication tensor T of order (n,m,p) has rank r
    # if T = sum_{k=1}^{r} u_k v_k w_k where u_k is in R^n, v_k in R^m, w_k in R^p.
    # Each term u_k v_k w_k corresponds to: C[i,j] += u_k[i] * v_k[j] * w_k[?]
    # Actually the standard form: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where the tensor T[i,j,k] = delta (the multiplication tensor).
    # For matrix mult: C = AB, C[i,j] = sum_k A[i,k]*B[k,j].
    # The tensor is T[i,k,j] = delta (it's 1 when the indices match appropriately).
    # Rank-1 decomposition: T = sum_r U[i,r]*V[k,r]*W[j,r]
    # where U has shape (n*m, r) indexed by (i,k), V has shape (m*p, r) indexed by (k,j),
    # W has shape (p*n, r) indexed by (j,i) per the task's trace convention.

    # Wait, the task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # And the decomposition is: for each output C[i,j], C[i,j] = sum_r U[i*n+j, r] * ... 
    # No, let me re-read: "the output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed as W[k*n+i, r] meaning W's first index is k*n+i.
    # U is indexed as U[i*m+j, r] (i is row of A, j is col of A / row of B)
    # V is indexed as V[j*p+k, r] (j is row of B, k is col of B)
    # W is indexed as W[k*n+i, r] (k is col of B / col of C... wait)

    # Let me just use the schoolbook and verify, then try to find improvements.
    # Schoolbook: rank = n*m*p = 216.
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.

    # Now, can we find a lower-rank decomposition?
    # Known results: rank of 4x4 matrix mult tensor is 20 (Laderman).
    # rank of 2x2 is 7 (Strassen).
    # For 6x6, the exact rank is unknown but bounded.
    # Lower bound: n^2 + m^2 + p^2 - n - m - p + 1 = 36+36+36-6-6-6+1 = 91 (for n=m=p=6)
    # Upper bound from various constructions...

    # One approach: decompose 6x6 as a block matrix and use known results.
    # 6 = 2+4. So A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x4, A21 is 4x2, A22 is 4x4.
    # Similarly for B and C.
    # C11 = A11*B11 (2x2*2x2, rank 7 with Strassen)
    # C12 = A11*B12 + A12*B22 (2x2*2x4 + 2x4*4x4)
    # C21 = A21*B11 + A22*B21 (4x2*2x4 + 4x4*4x2)
    # C22 = A21*B12 + A22*B22 (4x2*2x4 + 4x4*4x4)
    # This gets complicated with rectangular matrices.

    # Let me try a simpler approach: just use the schoolbook but see if there's a
    # straightforward improvement. Given the time constraint, let me first ensure
    # correctness, then attempt improvements.

    # For now, let's do the schoolbook (rank 216) to ensure we have a valid baseline,
    # then try to improve.

    # Actually, let me try a 3x3 grid of 2x2 Strassen blocks.
    # The key insight: in the tensor decomposition, we can group terms.
    # For the 3x3 grid of 2x2 blocks, the block product C[i,j] = sum_k A[i,k]*B[k,j]
    # Each A[i,k]*B[k,j] is a 2x2*2x2 product.
    # In Strassen, A[i,k]*B[k,j] = sum of 7 "Strassen products" S_t(A[i,k], B[k,j])
    # where each S_t is a product of linear combinations.
    # In the tensor decomposition, S_t = (sum_a alpha_{t,a} * A_entry_a) * (sum_b beta_{t,b} * B_entry_b)
    # This contributes 4 rank-1 terms (one for each combination of A_entry and B_entry).
    # So total rank = 27 * 7 * 4 = 756? That's way more than 216.
    
    # I'm confusing myself. Let me be very careful.
    
    # The tensor T has entries T[(i,k), (k,j), (j,i)]... no.
    # Standard: for matrix multiplication C=AB where A is n x m, B is m x p, C is n x p.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor: T[i,k,j] = 1 (it's the identity tensor in the middle index)
    # Rank-1 decomposition: T = sum_r U_i * V_k * W_j where
    #   U has shape (n, r), V has shape (m, r), W has shape (p, r)
    #   T[i,k,j] = sum_r U[i,r] * V[k,r] * W[j,r]
    
    # But the task uses a different indexing:
    # U has shape (n*m, rank) - indexed by (i, j) where i is row of A, j is col of A
    # V has shape (m*p, rank) - indexed by (j, k) where j is row of B, k is col of B
    # W has shape (p*n, rank) - indexed by (k, i) per "W[k*n+i, r]"
    
    # So the decomposition is: for each (i, j, k), the contribution to C[i,k] from A[i,j]*B[j,k]
    # is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't look right either. Let me re-derive.
    
    # C = A * B, C is n x p. C[i,k] = sum_j A[i,j] * B[j,k].
    # We want: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_? V[?, r]) * ...
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # T[i,j,k] where the tensor represents the bilinear map (A,B) -> C
    # C[i,k] = sum_j A[i,j] B[j,k]
    # As a 3-tensor: T[i,j,k] = delta (identity in j-k pairing)
    # Rank-1: T = sum_r u_r v_r w_r where u_r in R^{n}, v_r in R^{m}, w_r in R^{p}
    # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
    # This means C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r u_r[i] v_r[j] w_r[k] A[i,j] B[j,k]
    # Hmm, that's not quite the right form.
    
    # The correct form: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) ... no.
    # Let me look at this differently.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * (something)
    
    # Actually I think the convention is:
    # For each rank-1 term r, we have:
    #   intermediate_r = (U_r · A_flat) * (V_r · B_flat)
    #   C += W_r * intermediate_r (outer product or something)
    
    # Let me just look at the schoolbook and understand the indexing.
    # Schoolbook: for each (i,j,k), one rank-1 term with:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1  
    #   W[k*n+i, col] = 1
    # All other entries in column col are 0.
    
    # So C[i',k'] = sum_{i,j,k} U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] * A[i,j] * B[j,k]
    # For a specific (i,j,k) with col:
    #   Only U[i*m+j, col] = 1, so i'=i, j=j (from U)
    #   V[j*p+k, col] = 1, so j=j, k=k (from V)
    #   W[k*n+i, col] = 1, so the output index is (k, i) meaning i'=i, k'=k
    #   Contribution: A[i,j] * B[j,k] to C[i,k]
    
    # So C[i',k'] = sum_{(i,j,k): i=i', k=k'} A[i,j] B[j,k] = sum_j A[i',j] B[j,k']
    # That's correct!
    
    # Great, so the convention is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) ... no.
    # C[i,k] = sum_j A[i,j] B[j,k] where the decomposition gives:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    # Wait, that has two sums over j. Let me check with schoolbook:
    # W[k*n+i, r] = 1 for r corresponding to (i,j,k), 0 otherwise.
    # sum_j U[i*m+j, r] A[i,j] = A[i,j] (since only U[i*m+j, r] = 1)
    # sum_j V[j*p+k, r] B[j,k] = B[j,k]
    # So C[i,k] = sum_r W[k*n+i, r] * A[i,j_r] * B[j_r, k] where j_r is the j for that r.
    # = sum_{j} A[i,j] B[j,k] (since each (i,j,k) has its own r)
    # Yes! That works.
    
    # So the decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # Now, for Strassen 2x2 within the 3x3 block structure:
    # Consider block positions (I, J, K) where I, J, K in {0,1,2} index the 3x3 grid.
    # Block C[I,K] = sum_J BlockA[I,J] * BlockB[J,K]
    # Each BlockA[I,J] is 2x2, BlockB[J,K] is 2x2.
    # BlockA[I,J] * BlockB[J,K] is a 2x2*2x2 product.
    # In Strassen, this = 7 products of linear combinations.
    # Each Strassen product: (sum_{a} alpha_a * BlockA_entry_a) * (sum_{b} beta_b * BlockB_entry_b)
    # In the tensor decomposition, this contributes 4 rank-1 terms (one per entry pair).
    # Total: 9 * 3 * 7 * 4 = 756. That's much worse than 216!
    
    # I see the issue: Strassen reduces the number of "multiplications" but each multiplication
    # is of linear combinations, which in the tensor rank sense costs 4 rank-1 terms each.
    # So Strassen for 2x2 gives 7*4 = 28 rank-1 terms vs 4*4 = 16 for schoolbook.
    # Strassen is WORSE for tensor rank!
    
    # The tensor rank of 2x2 matrix multiplication is 7, not 28.
    # How? Because in the tensor decomposition, we can share the linear combinations.
    # T_2x2 = sum_{r=1}^{7} u_r v_r w_r
    # where u_r, v_r, w_r are 2-dimensional vectors (or rather, the appropriate shapes).
    # For 2x2: U has shape (2*2, 7) = (4,7), V has shape (2*2, 7) = (4,7), W has shape (2*2, 7) = (4,7).
    # This is the standard Strassen tensor decomposition.
    
    # So for the 3x3 grid of 2x2 blocks:
    # Total rank = 9 * 3 * 7 = 189.
    # That's better than 216!
    
    # Let me verify: for each block product position (I,J,K), we use the 2x2 Strassen decomposition.
    # The 2x2 Strassen has rank 7. There are 3*3*3 = 27 block product positions.
    # Total rank = 27 * 7 = 189.
    
    # Let me implement this!
    
    # Strassen 2x2 tensor decomposition (rank 7):
    # For A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In tensor form, for a 2x2 block product at position (I,J,K):
    # The A block has entries A[I*2+0, J*2+0], A[I*2+0, J*2+1], A[I*2+1, J*2+0], A[I*2+1, J*2+1]
    # which are indexed as U[(I*2+0)*m + (J*2+0), r], etc.
    # Similarly for B and C.
    
    # Let me define the Strassen factors for a 2x2 block.
    # For the 2x2 case, n=m=p=2.
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)
    # W has shape (2*2, 7) = (4, 7)
    
    # U indices: (i, j) -> i*2+j where i in {0,1}, j in {0,1}
    # V indices: (j, k) -> j*2+k
    # W indices: (k, i) -> k*2+i
    
    # Strassen factors:
    # Term 1: M1 = (a11+a22)(b11+b22) -> C11 += 1, C22 += 1
    #   U: a11 -> (0,0), a22 -> (1,1) both coeff 1
    #   V: b11 -> (0,0), b22 -> (1,1) both coeff 1
    #   W: C11 -> (0,0) coeff 1, C22 -> (1,1) coeff 1
    #   But this is one "product" that contributes to two C entries.
    #   In the tensor decomposition, M1 is one rank-1 term that contributes to both C11 and C22.
    #   So W has two non-zero entries: W[0*2+0, r] = 1 and W[1*2+1, r] = 1.
    
    # Let me write out all 7 Strassen terms for 2x2:
    
    def strassen_2x2_terms():
        """Return the 7 rank-1 terms for 2x2 Strassen.
        Each term is (U_coeffs, V_coeffs, W_coeffs) where each is a dict mapping index -> coeff.
        U index: i*2+j (A entry), V index: j*2+k (B entry), W index: k*2+i (C entry)
        """
        terms = []
        # Term 1: (a11+a22)(b11+b22) -> C11 += 1, C22 += 1
        terms.append((
            {(0,0): 1, (1,1): 1},  # a11 + a22
            {(0,0): 1, (1,1): 1},  # b11 + b22
            {(0,0): 1, (1,1): 1}   # C11 + C22
        ))
        # Term 2: (a21+a22)b11 -> C21 += 1, C22 += -1
        terms.append((
            {(1,0): 1, (1,1): 1},  # a21 + a22
            {(0,0): 1},             # b11
            {(0,1): 1, (1,1): -1}   # C21 - C22
        ))
        # Term 3: a11(b12-b22) -> C12 += 1, C22 += 1
        terms.append((
            {(0,0): 1},             # a11
            {(0,1): 1, (1,1): -1},  # b12 - b22
            {(0,1): 1, (1,1): 1}    # C12 + C22
        ))
        # Term 4: a22(b21-b11) -> C11 += 1, C21 += 1
        terms.append((
            {(1,1): 1},             # a22
            {(1,0): 1, (0,0): -1},  # b21 - b11
            {(0,0): 1, (1,0): 1}    # C11 + C21
        ))
        # Term 5: (a11+a12)b22 -> C11 += -1, C12 += 1
        terms.append((
            {(0,0): 1, (0,1): 1},   # a11 + a12
            {(1,1): 1},             # b22
            {(0,0): -1, (0,1): 1}   # -C11 + C12
        ))
        # Term 6: (a21-a11)(b11+b12) -> C22 += 1
        terms.append((
            {(1,0): 1, (0,0): -1},  # a21 - a11
            {(0,0): 1, (0,1): 1},   # b11 + b12
            {(1,1): 1}              # C22
        ))
        # Term 7: (a12-a22)(b21+b22) -> C11 += 1
        terms.append((
            {(0,1): 1, (1,1): -1},  # a12 - a22
            {(1,0): 1, (1,1): 1},   # b21 + b22
            {(0,0): 1}              # C11
        ))
        return terms
    
    strassen_terms = strassen_2x2_terms()
    
    # Now build the full 6x6x6 decomposition.
    # For each block position (I, J, K) in the 3x3 grid, and each Strassen term t:
    #   For each (u_idx, u_coeff) in strassen_terms[t][0]:
    #     U[(I*2 + u_idx[0]) * m + (J*2 + u_idx[1]), col] += u_coeff
    #   Similarly for V and W.
    
    # But wait, each Strassen term is ONE rank-1 term in the 2x2 decomposition.
    # So for each (I,J,K) and each of the 7 Strassen terms, we get ONE rank-1 term in the 6x6 decomposition.
    # Total rank = 27 * 7 = 189.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                    for (ui, uj), uc in u_coeffs.items():
                        u_row = (I*2 + ui) * m + (J*2 + uj)
                        U_dict[(u_row, col)] = U_dict.get((u_row, col), 0) + uc
                    for (vj, vk), vc in v_coeffs.items():
                        v_row = (J*2 + vj) * p + (K*2 + vk)
                        V_dict[(v_row, col)] = V_dict.get((v_row, col), 0) + vc
                    for (wk, wi), wc in w_coeffs.items():
                        w_row = (K*2 + wk) * n + (I*2 + wi)
                        W_dict[(w_row, col)] = W_dict.get((w_row, col), 0) + wc
                    col += 1
    
    rank = col
    
    # Convert dicts to lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, c), val in U_dict.items():
        U[row][c] = val
    for (row, c), val in V_dict.items():
        V[row][c] = val
    for (row, c), val in W_dict.items():
        W[row][c] = val
    
    # Verify correctness with a random test
    import random
    import numpy as np
    
    random.seed(42)
    A_test = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
    B_test = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
    C_expected = A_test @ B_test
    
    # Compute C using the decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for r in range(rank):
                # sum_j U[i*m+j, r] * A[i,j]
                u_sum = 0
                for j in range(m):
                    u_sum += U[i*m+j][r] * A_test[i][j]
                # sum_j V[j*p+k, r] * B[j,k]
                v_sum = 0
                for j in range(m):
                    v_sum += V[j*p+k][r] * B_test[j][k]
                total += W[k*n+i][r] * u_sum * v_sum
            C_computed[i][k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook if verification fails
        terms = []
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(((i*m+j, 1), (j*p+k, 1), (k*n+i, 1)))
                    col += 1
        rank = col
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for (u_idx, u_val), (v_idx, v_val), (w_idx, w_val) in terms:
            U[u_idx][terms.index(((u_idx, u_val), (v_idx, v_val), (w_idx, w_val)))] = u_val
            V[v_idx][terms.index(((u_idx, u_val), (v_idx, v_val), (w_idx, w_val)))] = v_val
            W[w_idx][terms.index(((u_idx, u_val), (v_idx, v_val), (w_idx, w_val)))] = w_val
        # Simpler schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        rank = n*m*p
    else:
        pass  # Verification passed
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
