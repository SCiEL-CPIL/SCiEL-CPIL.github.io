"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition by applying 2x2 Strassen at the block level.
    Split 4x4 into 2x2 blocks of 2x2.
    This gives 7 * (rank of 2x2) = 7*7 = 49 terms for 4x4*4x4->4x4.
    
    Actually, for 4x4 * 4x4 -> 4x4:
    Split into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2.
    Then apply Strassen to get 7 block products, each of which is a 2x2*2x2->2x2.
    Each 2x2*2x2->2x2 uses 7 scalar products.
    Total: 7 * 7 = 49 scalar products.
    
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # For 4x4:
    # A is 4x4, split into 2x2 blocks:
    #   A11 = rows 0-1, cols 0-1
    #   A12 = rows 0-1, cols 2-3
    #   A21 = rows 2-3, cols 0-1
    #   A22 = rows 2-3, cols 2-3
    # B is 4x4, split similarly.
    # C is 4x4, split similarly.
    
    # U: shape (16, 49) - row index = i*4+j (i in 0..3, j in 0..3)
    # V: shape (16, 49) - row index = j*4+k
    # W: shape (16, 49) - row index = k*4+i
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Block indices for A: (bi, bj) where bi in {0,1}, bj in {0,1}
    # Block (bi, bj) covers rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # Local row in block: li in {0,1}, local col: lj in {0,1}
    # Global row index: bi*2 + li, global col index: bj*2 + lj
    # U row index = (bi*2+li)*4 + (bj*2+lj)
    
    # Similarly for B: block (bj, bk), V row index = (bj*2+lj)*4 + (bk*2+lk)
    # For C: block (bi, bk), W row index = (bk*2+lk)*4 + (bi*2+li)
    
    # Strassen products at block level:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    # Block product assignments:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For each Strassen product Ps, we have a 2x2*2x2->2x2 subproblem.
    # The subproblem uses U2, V2, W2 with 7 scalar terms.
    
    # Map: for product s (0..6), the block-level coefficients determine
    # which blocks of A, B, C are involved.
    
    # Let's define for each Strassen product s:
    #   a_blocks: list of (block_idx_in_A, sign)
    #   b_blocks: list of (block_idx_in_B, sign)
    #   c_blocks: list of (block_idx_in_C, sign)
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_defs = [
        # (a_blocks, b_blocks, c_blocks)
        # P1: (A11+A22)*(B11+B22) -> C11 += 1, C22 += 1
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # P2: (A21+A22)*B11 -> C21 += 1, C22 -= 1
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # P3: A11*(B12-B22) -> C12 += 1, C22 += 1
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # P4: A22*(B21-B11) -> C11 += 1, C21 += 1
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # P5: (A11+A12)*B22 -> C11 -= 1, C12 += 1
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # P6: (A21-A11)*(B11+B12) -> C22 += 1
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # P7: (A12-A22)*(B21+B22) -> C11 += 1
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        # For this Strassen product, we have a 2x2*2x2->2x2 subproblem
        # with 7 scalar terms (from U2, V2, W2).
        # The subproblem's "A" is a 2x2 matrix formed by combining blocks.
        
        # For each scalar term t in the 2x2 subproblem (t=0..6):
        # U2[local_a_row, t] gives coefficient for element in "A_sub"
        # V2[local_b_row, t] gives coefficient for element in "B_sub"
        # W2[local_c_row, t] gives coefficient for element in "C_sub"
        
        for t in range(7):
            col = s * 7 + t
            
            # For each element in the 2x2 sub-A matrix:
            # local_a_row = la_i * 2 + la_j where la_i, la_j in {0,1}
            # This corresponds to A_sub[la_i, la_j]
            # A_sub is formed from blocks of A.
            # The block at position (la_i, la_j) in the 2x2 block matrix
            # is A_{(bi, bj)} where bi, bj depend on la_i, la_j.
            
            # Wait, I need to think about this more carefully.
            # The 2x2 Strassen operates on block matrices.
            # A_sub is a 2x2 matrix where each entry is a 2x2 block.
            # So A_sub[0,0] = A11, A_sub[0,1] = A12, A_sub[1,0] = A21, A_sub[1,1] = A22.
            
            # In the scalar 2x2*2x2->2x2 decomposition, U2[la_i*2+la_j, t] is the
            # coefficient for A_sub[la_i, la_j] in product t.
            # But A_sub[la_i, la_j] is itself a 2x2 block!
            # So we need to expand this.
            
            # Hmm, this is getting complicated. Let me reconsider.
            
            # Actually, the standard approach: when we say "apply Strassen to 4x4",
            # we're treating the 4x4 as a 2x2 block matrix where each block is 2x2.
            # The 7 Strassen products each involve 2x2*2x2->2x2 multiplications.
            # Each such multiplication can be done with 7 scalar products (using Strassen again)
            # or with 4 scalar products (schoolbook).
            
            # For the 4x4 case, if we use Strassen recursively:
            # 7 block products, each 2x2*2x2->2x2 with 7 scalar products = 49 total.
            
            # Let me restructure. The key insight:
            # A_sub[la_i, la_j] is a 2x2 block. When U2[la_i*2+la_j, t] = coeff,
            # it means that in product t, we multiply coeff * A_sub[la_i, la_j] * B_sub[mb_i, mb_j]
            # and add to C_sub[nc_i, nc_j].
            
            # But A_sub[la_i, la_j] * B_sub[mb_i, mb_j] is a 2x2*2x2->2x2 product.
            # We decompose this using U2, V2, W2.
            
            # So for each Strassen product s and each scalar term t in the 2x2 subproblem:
            # For each (la_i, la_j) with U2[la_i*2+la_j, t] != 0:
            #   For each (mb_i, mb_j) with V2[mb_i*2+mb_j, t] != 0:
            #     For each (nc_i, nc_j) with W2[nc_i*2+nc_j, t] != 0:
            #       This contributes to the overall product.
            
            # Wait, I'm overcomplicating. Let me think differently.
            
            # The 4x4*4x4->4x4 tensor can be decomposed by:
            # 1. Split into 2x2 blocks: 4 blocks in A, 4 in B, 4 in C.
            # 2. Apply Strassen at block level: 7 block-level products.
            # 3. Each block-level product is a 2x2*2x2->2x2, decomposed with 7 scalar products.
            
            # So total rank = 7 * 7 = 49.
            
            # For Strassen product s, the block-level formula is:
            #   prod_s = sum_{(bi,bj) in a_blocks} sign_a * A_{bi,bj} * sum_{(bk,bl) in b_blocks} sign_b * B_{bk,bl}
            #   Wait no, that's not right either.
            
            # Let me be very explicit. For P1 = (A11+A22)*(B11+B22):
            # This is a 2x2*2x2->2x2 where:
            #   A_sub = [[A11, 0], [A22, 0]]... no that's wrong.
            
            # Actually (A11+A22) is a 2x2 matrix (sum of two 2x2 blocks).
            # (B11+B22) is a 2x2 matrix.
            # Their product is a 2x2 matrix.
            # So we're multiplying two 2x2 matrices, which takes 7 scalar products via Strassen.
            
            # So for P1, define:
            #   A_P1 = A11 + A22 (a 2x2 matrix)
            #   B_P1 = B11 + B22 (a 2x2 matrix)
            #   C_P1 = A_P1 * B_P1 (a 2x2 matrix)
            #   C11 += C_P1, C22 += C_P1
            
            # To decompose C_P1 = A_P1 * B_P1 using Strassen (7 scalar products):
            # A_P1 is a 2x2 matrix. Let's write A_P1 = [[p11, p12], [p21, p22]]
            # where p11 = A11[0,0] + A22[0,0], p12 = A11[0,1] + A22[0,1], etc.
            
            # Similarly B_P1 = [[q11, q12], [q21, q22]]
            
            # Then apply the 2x2 Strassen to get 7 scalar products.
            
            # So the overall decomposition:
            # For each Strassen product s (0..6):
            #   Define A_s and B_s as 2x2 matrices (linear combinations of blocks).
            #   Decompose A_s * B_s using 2x2 Strassen (7 scalar products).
            #   Add the result to appropriate C blocks.
            
            # Let me implement this properly.
            
            pass  # This approach is getting too complex inline. Let me restart.
    
    # Let me use a cleaner approach.
    # I'll build the 4x4 decomposition by explicitly tracking all scalar terms.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # For each Strassen product s, define the 2x2 matrices A_s and B_s
    # as linear combinations of the 2x2 blocks.
    
    # A_s is 2x2, B_s is 2x2.
    # A_s[li, lj] = sum over blocks (bi,bj) of coeff * A_{bi,bj}[li, lj]
    # B_s[li, lj] = sum over blocks (bk,bl) of coeff * B_{bk,bl}[li, lj]
    
    # Then C_s = A_s * B_s is 2x2.
    # C_s[li, lj] = sum_{lk} A_s[li, lk] * B_s[lk, lj]
    
    # And C_{ci, cj} += coeff_c * C_s for each (ci,cj) in c_blocks.
    
    # To decompose C_s = A_s * B_s (2x2*2x2->2x2) using Strassen:
    # Use the 7 scalar products from _build_strassen_2x2().
    
    # Let me define the block linear combinations for each Strassen product.
    
    # For A_s[li, lj]:
    #   A_s[li, lj] = sum_{(bi,bj) in a_blocks} sign_a * A_{bi,bj}[li, lj]
    #   A_{bi,bj}[li, lj] = A[bi*2+li, bj*2+lj]
    #   So U row index = (bi*2+li)*4 + (bj*2+lj)
    
    # For B_s[li, lj]:
    #   B_s[li, lj] = sum_{(bk,bl) in b_blocks} sign_b * B_{bk,bl}[li, lj]
    #   B_{bk,bl}[li, lj] = B[bk*2+li, bl*2+lj]
    #   So V row index = (bk*2+li)*4 + (bl*2+lj)
    
    # For C_s[li, lj]:
    #   C_s[li, lj] = sum_{lk} A_s[li, lk] * B_s[lk, lj]
    #   C_{ci,cj}[li, lj] += coeff_c * C_s[li, lj]
    #   C_{ci,cj}[li, lj] = C[ci*2+li, cj*2+lj]
    #   So W row index = (cj*2+lj)*4 + (ci*2+li)
    
    # Now, for the 2x2*2x2->2x2 decomposition of C_s = A_s * B_s:
    # Using Strassen, we have 7 scalar products.
    # For scalar product t (0..6):
    #   term_t = (sum_{la} U2[la*2+lb, t] * A_s[la, lb]) * (sum_{mb} V2[mb*2+mc, t] * B_s[mb, mc])
    #   C_s[nc, nd] += W2[nc*2+nd, t] * term_t
    
    # Expanding:
    #   term_t = (sum_{la,lb} U2[la*2+lb, t] * sum_{(bi,bj)} sign_a * A[bi*2+la, bj*2+lb])
    #            * (sum_{mb,mc} V2[mb*2+mc, t] * sum_{(bk,bl)} sign_b * B[bk*2+mb, bl*2+mc])
    
    # This is a product of two linear combinations, which is not a single scalar product.
    # I need to be more careful.
    
    # Actually, the 2x2 Strassen decomposition says:
    #   C_s[nc, nd] = sum_t W2[nc*2+nd, t] * P_t
    #   where P_t = (sum_{la,lb} U2[la*2+lb, t] * A_s[la, lb]) * (sum_{mb,mc} V2[mb*2+mc, t] * B_s[mb, mc])
    
    # But P_t is a product of two sums, which is not a single scalar multiplication.
    # To make this into a tensor decomposition, I need to expand P_t as a sum of products.
    
    # Hmm, this is the key issue. The Strassen decomposition of 2x2*2x2->2x2
    # already gives a tensor decomposition with 7 terms. When I apply it at the block level,
    # each "scalar" in the 2x2 problem is actually a 2x2 block, and the product of two
    # "scalars" (blocks) is itself a 2x2*2x2->2x2 product.
    
    # So the correct approach is:
    # For each Strassen product s at the 4x4 level:
    #   A_s and B_s are 2x2 matrices of 2x2 blocks.
    #   C_s = A_s * B_s is a 2x2 matrix of 2x2 blocks.
    #   To compute C_s, I need to do 4 block multiplications (schoolbook) or
    #   7 block multiplications (Strassen).
    
    # If I use schoolbook for the inner 2x2*2x2->2x2:
    #   C_s[li, lj] = sum_{lk} A_s[li, lk] * B_s[lk, lj]
    #   Each A_s[li, lk] * B_s[lk, lj] is a 2x2*2x2->2x2 product.
    #   Using schoolbook, that's 4 scalar products.
    #   So total for one Strassen product: 4 * 4 = 16 scalar products.
    #   Total for 4x4: 7 * 16 = 112.
    
    # If I use Strassen for the inner 2x2*2x2->2x2:
    #   Each A_s[li, lk] * B_s[lk, lj] is a 2x2*2x2->2x2 product.
    #   Using Strassen, that's 7 scalar products.
    #   So total for one Strassen product: 4 * 7 = 28 scalar products.
    #   Total for 4x4: 7 * 28 = 196. That's worse!
    
    # Wait, that can't be right. Let me reconsider.
    
    # Oh, I see the issue. When I say "apply Strassen to 4x4", I mean:
    #   Treat 4x4 as 2x2 blocks of 2x2.
    #   The 7 Strassen products each involve multiplying 2x2 block matrices.
    #   But each "2x2 block matrix multiplication" is itself a 2x2*2x2->2x2 problem
    #   where the "scalars" are 2x2 blocks.
    
    # No wait, that's not right either. Let me think again.
    
    # Standard Strassen for 4x4:
    #   Split 4x4 into 2x2 blocks (each block is 2x2).
    #   Apply Strassen's 7 formulas at the block level.
    #   Each formula involves additions/subtractions of 2x2 blocks and one 2x2*2x2->2x2 multiplication.
    #   The 2x2*2x2->2x2 multiplication can be done with:
    #     - 4 scalar products (schoolbook)
    #     - 7 scalar products (Strassen)
    
    # If I use schoolbook for the inner multiplications:
    #   7 outer products * 4 inner scalar products = 28 total.
    #   But wait, the inner "2x2*2x2->2x2" where scalars are 2x2 blocks...
    #   No, the blocks are 2x2 matrices, and multiplying two 2x2 matrices requires
    #   4 scalar (element-wise) multiplications if the elements are scalars.
    #   But here the "elements" of the block matrix are 2x2 blocks, not scalars.
    
    # I think I'm confusing myself. Let me start from scratch.
    
    # The matrix multiplication tensor for 4x4*4x4->4x4 has rank at most 64 (schoolbook).
    # Strassen's algorithm reduces this to at most 28 for 4x4 (if we use Strassen recursively
    # with 2x2 base case using 7 scalar products):
    #   4x4 = 2x2 blocks of 2x2.
    #   7 Strassen products at block level.
    #   Each product is a 2x2*2x2->2x2 where the "scalars" are 2x2 blocks.
    #   To compute a 2x2*2x2->2x2 where scalars are 2x2 blocks, we need to compute
    #   4 block multiplications (schoolbook), each of which is a 2x2*2x2->2x2 with scalar entries.
    #   Wait, no. If the "scalars" are 2x2 blocks, then a "scalar multiplication" is
    #   actually a 2x2*2x2->2x2 block multiplication.
    
    # I think the correct way to think about it:
    #   The 4x4*4x4->4x4 problem can be seen as a 2x2*2x2->2x2 problem where
    #   each "entry" is a 2x2 block.
    #   Strassen reduces the number of "block multiplications" from 4 to 7.
    #   But each "block multiplication" is a 2x2*2x2->2x2 problem.
    #   If we solve each block multiplication with schoolbook (4 scalar products),
    #   total = 7 * 4 = 28.
    #   If we solve each block multiplication with Strassen (7 scalar products),
    #   total = 7 * 7 = 49.
    
    # So the rank for 4x4 using Strassen with schoolbook base is 28.
    # The rank for 4x4 using Strassen recursively is 49.
    
    # 28 < 49, so using schoolbook for the inner 2x2 is better!
    
    # Let me implement the 4x4 with Strassen at top level and schoolbook at inner level.
    
    # For each Strassen product s:
    #   A_s = sum_{(bi,bj)} sign_a * A_{bi,bj}  (a 2x2 matrix)
    #   B_s = sum_{(bk,bl)} sign_b * B_{bk,bl}  (a 2x2 matrix)
    #   C_s = A_s * B_s  (2x2 matrix, schoolbook: 4 terms)
    #   For each (li, lk, lj): C_s[li, lj] += A_s[li, lk] * B_s[lk, lj]
    #   For each (ci,cj) in c_blocks: C_{ci,cj} += sign_c * C_s
    
    # So for each s, and for each (li, lk, lj) in {0,1}^3:
    #   A_s[li, lk] = sum_{(bi,bj) in a_blocks} sign_a * A[bi*2+li, bj*2+lk]
    #   B_s[lk, lj] = sum_{(bk,bl) in b_blocks} sign_b * B[bk*2+lk, bl*2+lj]
    #   Product: A_s[li, lk] * B_s[lk, lj]
    #   This is a product of two linear combinations, which expands to a sum of products.
    
    # To make this a tensor decomposition, each "term" must be a single product
    # A[i,j] * B[j',k] * W[k',i'].
    
    # So I need to expand:
    #   (sum_{(bi,bj)} sign_a * A[bi*2+li, bj*2+lk]) * (sum_{(bk,bl)} sign_b * B[bk*2+lk, bl*2+lj])
    #   = sum_{(bi,bj), (bk,bl)} sign_a * sign_b * A[bi*2+li, bj*2+lk] * B[bk*2+lk, bl*2+lj]
    
    # Each such product contributes to:
    #   C[ci*2+li, cj*2+lj] += sign_c * A[bi*2+li, bj*2+lk] * B[bk*2+lk, bl*2+lj]
    
    # So for each (s, li, lk, lj, (bi,bj), (bk,bl)):
    #   U[(bi*2+li)*4 + (bj*2+lk)] += sign_a
    #   V[(bk*2+lk)*4 + (bl*2+lj)] += sign_b
    #   W[(cj*2+lj)*4 + (ci*2+li)] += sign_c
    #   for each (ci,cj) in c_blocks.
    
    # But this creates multiple terms that may share the same (U_row, V_row, W_row) triple,
    # which can be combined. However, for a valid tensor decomposition, each term is
    # a rank-1 tensor, and we can have multiple terms with the same U and V rows but
    # different W rows, etc.
    
    # The total number of terms is:
    #   For each s (7): for each (li, lk, lj) (8): for each (bi,bj) in a_blocks:
    #   for each (bk,bl) in b_blocks: for each (ci,cj) in c_blocks.
    
    # This could be large. Let me count for P1:
    #   a_blocks: 2, b_blocks: 2, c_blocks: 2
    #   Terms: 8 * 2 * 2 * 2 = 64 for P1 alone.
    #   Total for all 7: could be up to 7 * 8 * 2 * 2 * 2 = 448. That's more than 64!
    
    # This is because I'm not combining like terms. Many of these terms will have
    # the same (U_row, V_row) pair and can be combined by summing the W coefficients.
    
    # Actually, for a given (s, li, lk, lj, bi, bj, bk, bl), the U_row and V_row are fixed.
    # The W_row depends on (ci, cj). So for a fixed (U_row, V_row), there may be multiple
    # W_rows with different coefficients.
    
    # In a tensor decomposition, each rank-1 term is U_col * V_col * W_col.
    # If two terms have the same U_col and V_col but different W_cols, they can be
    # combined into a single term with W_col being the sum.
    
    # But if they have different U_cols or V_cols, they must be separate terms.
    
    # Let me just build the decomposition and count the rank.
    
    # I'll use a dictionary to accumulate coefficients.
    # Key: (U_row, V_row, W_row), Value: coefficient.
    
    terms = {}
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    for (bi, bj), sign_a in a_blocks:
                        u_row = (bi*2+li)*4 + (bj*2+lk)
                        for (bk, bl), sign_b in b_blocks:
                            v_row = (bk*2+lk)*4 + (bl*2+lj)
                            for (ci, cj), sign_c in c_blocks:
                                w_row = (cj*2+lj)*4 + (ci*2+li)
                                coeff = sign_a * sign_b * sign_c
                                key = (u_row, v_row, w_row)
                                terms[key] = terms.get(key, 0) + coeff
    
    # Now, build U, V, W from the terms.
    # Each unique (u_row, v_row, w_row) with non-zero coefficient is a rank-1 term.
    # But we can combine terms with the same (u_row, v_row) by summing W coefficients.
    # Actually, in a tensor decomposition, each term is a rank-1 tensor.
    # The rank is the number of terms. We can't combine terms with different W_rows
    # into a single term unless we're willing to change the structure.
    
    # Wait, actually we CAN. If we have two terms:
    #   U1 * V1 * W1 + U1 * V1 * W2 = U1 * V1 * (W1 + W2)
    # So terms with the same U and V columns can be combined.
    
    # Let me restructure: group by (u_row, v_row).
    # For each (u_row, v_row), we have a set of (w_row, coeff) pairs.
    # We can create one term per (u_row, v_row, w_row) group, but that doesn't reduce rank.
    
    # Actually, the rank is the minimum number of rank-1 tensors needed.
    # If I have terms (u1, v1, w1, c1) and (u1, v1, w2, c2), I can write this as:
    #   c1 * u1 * v1 * w1 + c2 * u1 * v1 * w2
    # This is 2 rank-1 terms. I can't combine them into 1 unless w1 = w2.
    
    # So the rank is the number of non-zero (u_row, v_row, w_row) triples.
    
    # But wait, I can also factor out common U or V columns.
    # For example, if I have:
    #   u1 * v1 * w1 + u1 * v2 * w2
    # This is 2 terms. I can't combine them.
    
    # So the rank is just the number of non-zero triples.
    
    # Let me count.
    rank = sum(1 for v in terms.values() if v != 0)
    
    # Build U, V, W.
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for (u_row, v_row, w_row), coeff in terms.items():
        if coeff == 0:
            continue
        U[u_row][col] = coeff
        V[v_row][col] = 1
        W[w_row][col] = coeff
        col += 1
    
    # Trim if necessary
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        rank = col
    
    return U, V, W

def _build_strassen_4x5_5x6():
    """
    Build the decomposition for 4x5 * 5x6 -> 4x6 using Strassen-style blocking.
    
    Strategy: Split the 4x5 matrix into blocks and the 5x6 matrix into blocks,
    then use Strassen-like formulas to reduce the number of scalar products.
    
    For 4x5 * 5x6:
    - Split rows of A (4 rows) into 2 groups of 2.
    - Split cols of A (5 cols) into 2 groups: 2 and 3.
    - Split rows of B (5 rows) into 2 groups: 2 and 3.
    - Split cols of B (6 cols) into 2 groups of 3.
    
    This gives a 2x2 block structure where blocks have different sizes.
    Strassen's formulas assume square blocks, so this won't work directly.
    
    Alternative: Use a different blocking strategy.
    
    Let me try: Split 4x5 into top 2x5 and bottom 2x5.
    Split 5x6 into top 5x3 and bottom 5x3... no, that doesn't align.
    
    The issue is that 5 is not even, so we can't split it evenly.
    
    Let me try a different approach: use the known result that the rank of
    4x5*5x6 is at most 28 (from Strassen-type algorithms) or use a
    different decomposition.
    
    Actually, let me just use the schoolbook for now and try to optimize.
    Schoolbook rank = 4*5*6 = 120.
    
    Let me try a partial Strassen: split the 4 rows into 2+2, and handle
    the 5 and 6 dimensions with schoolbook.
    
    For 4x5 * 5x6:
    A = [[A1], [A2]] where A1 is 2x5, A2 is 2x5.
    B = [[B1, B2]] where B1 is 5x3, B2 is 5x3.
    C = [[C1, C2]] where C1 is 4x3, C2 is 4x3.
    
    C1 = A1*B1 + A2*B1 = (A1+A2)*B1
    C2 = A1*B2 + A2*B2 = (A1+A2)*B2
    
    So we need:
    P1 = (A1+A2)*B1 (2x5 * 5x3 -> 2x3, but we need 4x3)
    
    Wait, this doesn't work because A1 and A2 are 2x5, and their sum is 2x5,
    but we need to recover the 4x3 result from a 2x3 intermediate.
    
    Let me try:
    C1 = A1*B1 + A2*B1
    C2 = A1*B2 + A2*B2
    
    If I compute P1 = (A1+A2)*B1 and P2 = (A1+A2)*B2, I get 2x3 matrices,
    but I need 4x3. I can't recover A1*B1 and A2*B1 from (A1+A2)*B1 alone.
    
    I need additional products. For example:
    P1 = A1*B1
    P2 = A2*B1
    P3 = A1*B2
    P4 = A2*B2
    
    That's 4 products, each of rank 2*5*3 = 30, total 120. No improvement.
    
    Let me try Strassen-style:
    P1 = (A1+A2)*(B1+B2)
    P2 = A1*B2
    P3 = A2*B1
    
    Then:
    A1*B1 = ? I can't recover it from these.
    
    This is getting complicated. Let me just use the schoolbook decomposition
    for now, which gives rank 120, and see if I can find a better approach.
    
    Actually, let me try a different blocking:
    Split A (4x5) into 4 blocks of 1x5.
    Split B (5x6) into 6 blocks of 5x1.
    Split C (4x6) into 24 blocks of 1x1.
    
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    This is just the schoolbook with rank 120.
    
    Let me try to use the fact that 4 = 2*2 and see if I can apply Strassen
    to the 2x2 block structure in the row dimension.
    
    A = [[A11, A12, A13, A14, A15], [A21, A22, A23, A24, A25],
         [A31, A32, A33, A34, A35], [A41, A42, A43, A44, A45]]
    
    Split into top 2 rows and bottom 2 rows:
    A_top = 2x5, A_bot = 2x5.
    
    B = 5x6.
    
    C_top = A_top * B (2x6)
    C_bot = A_bot * B (2x6)
    
    C = [[C_top], [C_bot]]
    
    If I compute P1 = A_top * B and P2 = A_bot * B, that's 2 products,
    each of rank 2*5*6 = 60, total 120. No improvement.
    
    Let me try:
    P1 = (A_top + A_bot) * B
    P2 = A_top * B
    
    Then A_bot * B = P1 - P2. So I still need 2 products.
    
    It seems like for non-square matrices, Strassen's algorithm doesn't directly apply.
    
    Let me just use the schoolbook decomposition for the 4x5*5x6 case.
    """
    n, m, p = 4, 5, 6
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
    return U, V, W

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try to build a Strassen-based decomposition for 4x5*5x6.
    # For now, use schoolbook as fallback.
    
    U, V, W = _build_strassen_4x5_5x6()
    
    if time.time() > deadline:
        # Return schoolbook as fallback
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
