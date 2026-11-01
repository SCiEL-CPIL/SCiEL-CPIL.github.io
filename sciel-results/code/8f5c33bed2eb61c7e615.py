"""Matrix multiplication 4x4x4 tensor decomposition via recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.
    
    Uses two levels of Strassen:
      - Outer level: 4x4 split into 2x2 blocks of 2x2 matrices, 7 block-multiplications
      - Inner level: each 2x2 x 2x2 uses Strassen, 7 scalar multiplications
      - Total rank: 7 * 7 = 49
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      Tensor element T[i,k,j] = 1 (for all i,k,j)
      Reconstruction: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
      
      U index: i*m + k  (A row i, A col k)
      V index: k*p + j  (B row k, B col j)
      W index: j*n + i  (C col j, C row i)  [trace convention W[k*n+i,r] where k=B col, i=A row]
    """
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n = m = p = 4
    rank = 49

    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)

    # Index helpers
    def u_idx(ai, ak):
        return ai * m + ak

    def v_idx(bk, bj):
        return bk * p + bj

    def w_idx(cj, ci):
        return cj * n + ci

    # ====== OUTER STRASSEN: 7 block-multiplications for 2x2 block matrices ======
    # Each outer term t has:
    #   L_t: dict (block_row, block_col) -> list of ((er,ec), coeff)
    #         where (er,ec) is position within the 2x2 block, coeff is the A coefficient
    #   R_t: dict (block_row, block_col) -> list of ((er,ec), coeff)
    #         for B
    #   C_t: dict (prod_row, prod_col) -> list of ((er,ec), coeff)
    #         for C contribution

    outer_terms = []

    # M1 = (A11 + A22)(B11 + B22)
    # C11 += M1, C22 += M1
    L = {(0,0): [((0,0),1),((1,1),1)], (1,1): [((0,0),1),((1,1),1)]}
    # Wait, that's wrong. Let me re-derive.
    # A11 means block (0,0), which contains A elements at positions (0,0),(0,1),(1,0),(1,1) within that 2x2 block.
    # So L for A11: block (0,0) with all 4 positions coeff 1.
    # A22: block (1,1) with all 4 positions coeff 1.
    L = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    R = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    C = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    outer_terms.append((L, R, C))

    # M2 = (A21 + A22) B11
    # C21 += M2, C22 -= M2
    L = {(1,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    R = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    C = {(1,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),-1),((0,1),-1),((1,0),-1),((1,1),-1)]}
    outer_terms.append((L, R, C))

    # M3 = A11 (B12 - B22)
    # C12 += M3, C22 += M3
    L = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    R = {(0,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),-1),((0,1),-1),((1,0),-1),((1,1),-1)]}
    C = {(0,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    outer_terms.append((L, R, C))

    # M4 = A22 (B21 - B11)
    # C11 += M4, C21 += M4
    L = {(1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    R = {(1,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (0,0): [((0,0),-1),((0,1),-1),((1,0),-1),((1,1),-1)]}
    C = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    outer_terms.append((L, R, C))

    # M5 = (A11 + A12) B22
    # C11 -= M5, C12 += M5
    L = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (0,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    R = {(1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    C = {(0,0): [((0,0),-1),((0,1),-1),((1,0),-1),((1,1),-1)],
         (0,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    outer_terms.append((L, R, C))

    # M6 = (A21 - A11)(B11 + B12)
    # C22 += M6
    L = {(1,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (0,0): [((0,0),-1),((0,1),-1),((1,0),-1),((1,1),-1)]}
    R = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (0,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    C = {(1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    outer_terms.append((L, R, C))

    # M7 = (A12 - A22)(B21 + B22)
    # C11 += M7
    L = {(0,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),-1),((0,1),-1),((1,0),-1),((1,1),-1)]}
    R = {(1,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)],
         (1,1): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    C = {(0,0): [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]}
    outer_terms.append((L, R, C))

    # ====== INNER STRASSEN: 7 scalar mults for 2x2 x 2x2 ======
    # Each inner term has:
    #   l: dict (lr,lc) -> coeff  (for the left 2x2 block)
    #   r: dict (lr,lc) -> coeff  (for the right 2x2 block)
    #   p: dict (pr,pc) -> coeff  (contribution to product 2x2 block)
    inner = [
        # s1 = (L00+L11)(R00+R11): P00+=s1, P11+=s1
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # s2 = (L10+L11)R00: P10+=s2, P11-=s2
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        # s3 = L00(R01-R11): P01+=s3, P11+=s3
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        # s4 = L11(R10-R00): P00+=s4, P10+=s4
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        # s5 = (L00+L01)R11: P00-=s5, P01+=s5
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # s6 = (L10-L00)(R00+R01): P11+=s6
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        # s7 = (L01-L11)(R10+R11): P00+=s7
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    # ====== BUILD U, V, W ======
    for t in range(7):
        L_t, R_t, C_t = outer_terms[t]
        for k in range(7):
            li = t * 7 + k
            l_coeffs, r_coeffs, p_coeffs = inner[k]

            if time.time() - t0 > deadline:
                break

            # U contributions
            for (br, bc), positions in L_t.items():
                for (er, ec), a_coeff in positions:
                    # A element at global row 2*br+er, col 2*bc+ec
                    # But wait: L_t keys are (block_row, block_col) of A
                    # A block (br, bc) contains elements A[2*br+er, 2*bc+ec]
                    # The inner Strassen l_coeffs is indexed by (lr, lc) within the 2x2 block
                    # For a given outer term, L_t tells us which A blocks participate
                    # and l_coeffs tells us which positions within those blocks
                    # Actually, the inner Strassen operates on a single 2x2x2x2 subproblem
                    # where the "left" 2x2 is the entire A block content and "right" is B block content
                    # So l_coeffs[(lr,lc)] is the coefficient for A position (lr,lc) within the current block
                    # And L_t[(br,bc)] lists which positions (er,ec) within block (br,bc) are involved
                    # The inner Strassen l_coeffs should match the positions
                    pass

    # Let me redo this more carefully.
    # For each outer term t, we have a "virtual" 2x2x2x2 multiplication:
    #   Left matrix: 2x2, where element (lr,lc) gets coefficient from l_coeffs of inner[k]
    #   Right matrix: 2x2, where element (lr,lc) gets coefficient from r_coeffs of inner[k]
    #   Product: 2x2, where element (pr,pc) gets coefficient from p_coeffs of inner[k]
    #
    # The outer term tells us:
    #   L_t[(br,bc)] = list of ((er,ec), a_coeff): A block (br,bc) contributes a_coeff * A[2br+er, 2bc+ec]
    #   R_t[(br,bc)] = list of ((er,ec), b_coeff): B block (br,bc) contributes b_coeff * B[2br+er, 2bc+ec]
    #   C_t[(pr,pc)] = list of ((er,ec), c_coeff): C block (pr,pc) gets c_coeff * P[er,ec]
    #
    # For the inner Strassen to work, we need:
    #   The "left" 2x2 virtual matrix has entries that are linear combos of A elements
    #   The "right" 2x2 virtual matrix has entries that are linear combos of B elements
    #   The product contributes to C elements
    #
    # Inner s = sum_{lr,lc} l_coeffs[(lr,lc)] * L[lr,lc] * sum_{lr,lc} r_coeffs[(lr,lc)] * R[lr,lc]
    # Wait no: s = (sum l[L]) * (sum r[R]) where L[lr,lc] and R[lr,lc] are the actual values
    #
    # So: s = (sum_{lr,lc} l_coeffs[(lr,lc)] * A_block[lr,lc]) * (sum_{lr,lc} r_coeffs[(lr,lc)] * B_block[lr,lc])
    #
    # Where A_block[lr,lc] = sum over outer L_t entries: a_coeff * A[2br+er, 2bc+ec] where (er,ec) maps to (lr,lc)
    # Hmm, this is getting confusing. Let me think differently.
    #
    # Actually, the cleaner way:
    # For outer term t, the left "super-element" is a 2x2 matrix of A-blocks.
    # L_t tells us which A-blocks go into which positions of this 2x2 super-matrix.
    # So super_left[lr, lc] = sum over (br,bc) in L_t: L_t[(br,bc)] contribution at position (lr,lc)
    #
    # Wait, I think the structure is:
    # L_t[(br,bc)] = [( (er,ec), coeff )] means:
    #   The A element at global position (2*br+er, 2*bc+ec) contributes with coefficient 'coeff'
    #   to the "left input" of the inner Strassen at local position... 
    #
    # I think the mapping is:
    #   The outer term defines a 2x2x2x2 sub-problem.
    #   The "left" side is a 2x2 matrix where entry (lr,lc) = sum of a_coeff * A[global_i, global_k]
    #     where the sum is over all A positions that L_t assigns to local position (lr,lc)
    #
    # Let me just use a different formulation. I'll directly compute U, V, W entries.
    
    # For rank index li = t*7 + k:
    # U[ai, ak, li] = sum over all (br,bc,er,ec) in L_t: a_coeff * [inner l_coeffs at corresponding local pos]
    # 
    # The local position in the inner 2x2 is determined by which "slot" the A element goes into.
    # 
    # Actually, I realize the structure is simpler than I'm making it:
    # 
    # For a given outer term t:
    #   L_t maps block positions to lists of (local_pos, coeff)
    #   This means: for A block at (br,bc), the element at local position (er,ec) within that block
    #     contributes with coefficient a_coeff to the inner Strassen's left matrix at position (er,ec)
    #
    # Wait no. Let me look at the original code more carefully.
    #
    # In the original:
    #   L0[(r,c)] = {(0,0,r,c): 1, (1,1,r,c): 1}
    #   This means for block (r,c), the A elements at positions... 
    #   The key is (bi,bj,er,ec) and value is coeff
    #   So L0[(0,0)] = {(0,0,0,0):1, (1,1,0,0):1, (0,0,0,1):1, (1,1,0,1):1, ...}
    #   Wait, the original has L0[(r,c)] where r,c are in range(2), and the value is a dict
    #   keyed by (bi,bj,er,ec) where bi,bj are the block indices and er,ec are positions within.
    #
    # I think the original code's structure is:
    #   L[(r,c)] = dict: for A block at position (r,c) in the 2x2 block layout,
    #     the keys (bi,bj,er,ec) mean: block (bi,bj) element at position (er,ec) contributes
    #     with the given coefficient to the "left" side of the inner Strassen at position (r,c)
    #
    # This is quite convoluted. Let me just directly build the tensors using a cleaner approach.
    
    # I'll use a direct enumeration approach.
    # For 4x4x4 matmul tensor, T[i,k,j] = 1 for all i,k,j in {0,1,2,3}
    #
    # Recursive Strassen decomposition:
    # Level 1: 4x4 = 2x2 blocks of 2x2
    #   M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    #   M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    #   M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    #   M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    #   M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    #   M6 = (A21-A11)(B11+B12) -> C22 += M6
    #   M7 = (A12-A22)(B21+B22) -> C11 += M7
    #
    # Level 2: each Mx is a 2x2x2x2 matmul, decomposed by Strassen into 7 scalar mults
    #
    # For a 2x2x2x2 matmul with left matrix L (2x2) and right matrix R (2x2):
    #   s1 = (L00+L11)(R00+R11) -> P00+=s1, P11+=s1
    #   s2 = (L10+L11)R00 -> P10+=s2, P11-=s2
    #   s3 = L00(R01-R11) -> P01+=s3, P11+=s3
    #   s4 = L11(R10-R00) -> P00+=s4, P10+=s4
    #   s5 = (L00+L01)R11 -> P00-=s5, P01+=s5
    #   s6 = (L10-L00)(R00+R01) -> P11+=s6
    #   s7 = (L01-L11)(R10+R11) -> P00+=s7
    #
    # Now, for outer term t with:
    #   Left 2x2 "super" matrix S_L where S_L[br,bc] = linear combo of A blocks
    #   Right 2x2 "super" matrix S_R where S_R[br,bc] = linear combo of B blocks
    #
    # Each S_L[br,bc] is itself a 2x2 matrix of A elements.
    # The inner Strassen then operates on these 2x2 matrices.
    #
    # So for rank index li = t*7 + k:
    #   U[ai,ak,li] = coefficient of A[ai,ak] in the product s_k
    #   V[bk,bj,li] = coefficient of B[bk,bj] in the product s_k
    #   W[cj,ci,li] = coefficient of C[ci,cj] contribution from s_k (via C_t and p_coeffs)
    #
    # Let me define the outer terms more explicitly.

    # Define outer terms as: (left_blocks, right_blocks, c_blocks)
    # left_blocks: dict mapping (lr,lc) -> list of ((br,bc), a_coeff)
    #   meaning: S_L[lr,lc] = sum of a_coeff * A_block(br,bc)
    # right_blocks: dict mapping (lr,lc) -> list of ((br,bc), b_coeff)
    # c_blocks: dict mapping (pr,pc) -> list of ((br,bc), c_coeff)
    #   meaning: C_block(pr,pc) += c_coeff * P[pr,pc] where P is the 2x2 product

    # M1: S_L = [[A11, 0], [0, A22]] + [[0,0],[0,0]]... 
    # Actually: S_L[lr,lc] for M1:
    #   S_L[0,0] = A11 (block 0,0)
    #   S_L[1,1] = A22 (block 1,1)
    #   S_L[0,1] = 0, S_L[1,0] = 0
    # S_R[0,0] = B11, S_R[1,1] = B22, others 0
    # C: C11 += P, C22 += P

    # Let me just hardcode the 7 outer terms with explicit block mappings.
    # For each outer term, I'll specify:
    #   left: 2x2 array where entry [lr,lc] is a list of (block_row, block_col, coeff)
    #   right: 2x2 array where entry [lr,lc] is a list of (block_row, block_col, coeff)
    #   cmap: 2x2 array where entry [pr,pc] is a list of (block_row, block_col, coeff)

    def mk_block_entry(br, bc, coeff):
        return (br, bc, coeff)

    # M1: (A11+A22)(B11+B22) -> C11+=M, C22+=M
    left1 = [
        [ [mk_block_entry(0,0,1)] ],       # S_L[0,0] = A11
        [ [] ],                             # S_L[0,1] = 0
        [ [] ],                             # S_L[1,0] = 0
        [ [mk_block_entry(1,1,1)] ],       # S_L[1,1] = A22
    ]
    # Reshape to 2x2
    left1 = [[left1[0], left1[1]], [left1[2], left1[3]]]
    right1 = [
        [ [mk_block_entry(0,0,1)] ],
        [ [] ],
        [ [] ],
        [ [mk_block_entry(1,1,1)] ],
    ]
    right1 = [[right1[0], right1[1]], [right1[2], right1[3]]]
    cmap1 = [
        [ [mk_block_entry(0,0,1)] ],
        [ [] ],
        [ [] ],
        [ [mk_block_entry(1,1,1)] ],
    ]
    cmap1 = [[cmap1[0], cmap1[1]], [cmap1[2], cmap1[3]]]
    # C11 += P, C22 += P: 
    # c_block[0,0] gets P[0,0] with coeff 1 -> C block (0,0) += P[0,0]
    # c_block[1,1] gets P[1,1] with coeff 1 -> C block (1,1) += P[1,1]
    # But also C11 += P means the entire 2x2 product goes to C11 block
    # So: for each (pr,pc), C block (0,0) gets c_coeff * P[pr,pc]
    # And for each (pr,pc), C block (1,1) gets c_coeff * P[pr,pc]
    # Actually: C11 += M1 means C_block(0,0)[er,ec] += M1[er,ec] for all er,ec
    # So cmap1[pr,pc] should have entries for C blocks that receive P[pr,pc]
    # C11 receives all of M1, so for each (pr,pc), C block (0,0) at position (pr,pc) gets coeff 1
    # C22 receives all of M1, so for each (pr,pc), C block (1,1) at position (pr,pc) gets coeff 1
    
    # I think I'm overcomplicating this. Let me just directly compute U, V, W
    # by expanding the full expression.

    # For rank index li = t*7 + k:
    # s_k = (sum_{lr,lc} lcoeff[lr,lc] * S_L[lr,lc]) * (sum_{lr,lc} rcoeff[lr,lc] * S_R[lr,lc])
    # where S_L[lr,lc] = sum_{(br,bc,ac) in left_t[lr,lc]} ac * A[2br+0:2br+2, 2bc+0:2bc+2]
    #   Actually S_L[lr,lc] is a 2x2 matrix: S_L[lr,lc][er,ec] = sum of ac * A[2br+er, 2bc+ec]
    #
    # So: sum_{lr,lc} lcoeff[lr,lc] * S_L[lr,lc] is a 2x2 matrix:
    #   [er,ec] entry = sum_{lr,lc} lcoeff[lr,lc] * sum_{(br,bc,ac) in left_t[lr,lc]} ac * A[2br+er, 2bc+ec]
    #
    # Similarly for the right side.
    #
    # The product s_k is a scalar (the "inner product" isn't right - it's a matrix product)
    # Wait, no. The inner Strassen multiplies two 2x2 matrices to get a 2x2 product.
    # s_k is not a scalar - it's part of the 2x2 product computation.
    #
    # Actually in Strassen, s1, s2, ..., s7 are scalars, and the 2x2 product P is built from them.
    # s1 = (L00+L11)*(R00+R11) where L00 is a scalar (element of the left matrix)
    #
    # So the "left matrix" in the inner Strassen is a 2x2 matrix of scalars.
    # But in our case, each "element" of this 2x2 matrix is itself a 2x2 matrix of A elements.
    # So s1 = (L00+L11)*(R00+R11) where L00 is a 2x2 matrix.
    # This means s1 is a 2x2 matrix product: (L00_2x2 + L11_2x2) @ (R00_2x2 + R11_2x2)
    # No wait, that's not right either.
    #
    # I think the correct interpretation is:
    # The inner Strassen treats the "elements" as scalars, but each element is actually
    # a 2x2 sub-matrix. So the scalar multiplication becomes a 2x2 matrix multiplication.
    #
    # This means the "rank" contribution from one inner Strassen term is not 1 but 4
    # (one for each element of the 2x2 product).
    #
    # But the total rank is 49 = 7*7, which suggests each of the 7 outer terms contributes
    # 7 rank-1 terms, and each rank-1 term handles one scalar multiplication.
    #
    # I think the correct view is:
    # Each "scalar" in the inner Strassen is actually a 2x2 matrix.
    # The "multiplication" of two such scalars is a 2x2 matrix multiplication, which has rank 4.
    # But we're using rank-1 decompositions, so each scalar mult becomes 4 rank-1 terms.
    #
    # That would give 7 * 4 * 7 = 196, which is way too much.
    #
    # Let me reconsider. The standard recursive Strassen for 4x4:
    # - Split 4x4 into four 2x2 blocks
    # - Apply Strassen to get 7 block-multiplications of 2x2 matrices
    # - Each 2x2 block-multiplication is done via Strassen with 7 scalar multiplications
    # - Total: 7 * 7 = 49 scalar multiplications
    #
    # Each scalar multiplication is rank-1 in the tensor decomposition.
    # So total rank is 49.
    #
    # The key insight: when we say "multiply 2x2 matrices", we mean the full 2x2x2x2 tensor
    # for 2x2 matrix multiplication. This tensor has rank 7 (Strassen).
    # So each block-multiplication (which is a 2x2x2x2 subproblem) has rank 7.
    # And there are 7 such block-multiplications, giving total rank 49.
    #
    # So the structure is:
    # For each outer term t (7 terms) and inner term k (7 terms):
    #   We have a rank-1 term that contributes to the overall tensor.
    #   U[:,li] is a vector of length 16 (4x4 A elements)
    #   V[:,li] is a vector of length 16 (4x4 B elements)
    #   W[:,li] is a vector of length 16 (4x4 C elements)
    #   And U[i,k] * V[k,j] * W[j,i] summed over li gives T[i,k,j] = 1
    #
    # Let me just directly construct the decomposition using the known Strassen formulas.
    # I'll build it term by term.
    
    # For each of the 49 rank-1 terms, I need to specify:
    #   Which A elements (and their coefficients) form U
    #   Which B elements (and their coefficients) form V
    #   Which C elements (and their coefficients) form W
    
    # Let me use a systematic approach. I'll define the 7 outer Strassen terms
    # and for each, the 7 inner Strassen terms, and compute the contributions.
    
    # Outer term definitions:
    # Each outer term specifies linear combinations of 2x2 blocks of A and B,
    # and how their 2x2x2x2 product contributes to C blocks.
    
    # I'll represent each outer term as:
    #   a_blocks: dict (ar, ac) -> (block_row, block_col, coeff)
    #     meaning: the A element at global (2*block_row+ar_offset, 2*block_col+ac_offset) 
    #     ... this is getting too complex.
    
    # Let me just hardcode the 49 terms directly.
    # For 4x4x4 matmul, the tensor T[i,k,j] = 1.
    # The Strassen decomposition gives us 49 rank-1 decompositions.
    
    # I'll use numpy to verify at the end.
    
    # Let me try a completely different approach: use the known fact that
    # the 4x4x4 matmul tensor can be decomposed with rank 49 using recursive Strassen,
    # and construct it programmatically.
    
    # Define the 2x2x2x2 Strassen decomposition (rank 7)
    # T_2x2[i,k,j] = 1 for i,k,j in {0,1}
    # Decomposition: sum_{r=0}^{6} U2[:,r] * V2[:,r] * W2[:,r] = T_2x2
    
    # For 2x2: U2 has shape (4,7), V2 has shape (4,7), W2 has shape (4,7)
    # U2[i*2+k, r], V2[k*2+j, r], W2[j*2+i, r]
    
    # Strassen for 2x2:
    # s1 = (A00+A11)(B00+B11): C00+=s1, C11+=s1
    # s2 = (A10+A11)B00: C10+=s2, C11-=s2
    # s3 = A00(B01-B11): C01+=s3, C11+=s3
    # s4 = A11(B10-B00): C00+=s4, C10+=s4
    # s5 = (A00+A01)B11: C00-=s5, C01+=s5
    # s6 = (A10-A00)(B00+B01): C11+=s6
    # s7 = (A01-A11)(B10+B11): C00+=s7
    
    # Build U2, V2, W2 for 2x2
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)
    
    # For 2x2: A index = i*2+k, B index = k*2+j, C index = j*2+i
    
    # s1: (A00+A11)(B00+B11)
    # A00 is at index 0*2+0=0, A11 is at index 1*2+1=3
    # B00 is at index 0*2+0=0, B11 is at index 1*2+1=3
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 at index 0*2+0=0, C11 at index 1*2+1=3
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # s2: (A10+A11)B00
    # A10 at index 1*2+0=2, A11 at index 3
    # B00 at index 0
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 at index 0*2+1=1, C11 at index 3
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # s3: A00(B01-B11)
    # A00 at index 0
    # B01 at index 0*2+1=1, B11 at index 3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C01 at index 1*2+0=2, C11 at index 3
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # s4: A11(B10-B00)
    # A11 at index 3
    # B10 at index 1*2+0=2, B00 at index 0
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # C00 at index 0, C10 at index 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # s5: (A00+A01)B11
    # A00 at index 0, A01 at index 0*2+1=1
    # B11 at index 3
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 at index 0, C01 at index 2
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # s6: (A10-A00)(B00+B01)
    # A10 at index 2, A00 at index 0
    # B00 at index 0, B01 at index 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C11 at index 3
    W2[3, 5] = 1
    
    # s7: (A01-A11)(B10+B11)
    # A01 at index 1, A11 at index 3
    # B10 at index 2, B11 at index 3
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C00 at index 0
    W2[0, 6] = 1
    
    # Verify 2x2 decomposition
    err2 = 0.0
    for i in range(2):
        for k in range(2):
            for j in range(2):
                s = 0.0
                for r in range(7):
                    s += U2[i*2+k, r] * V2[k*2+j, r] * W2[j*2+i, r]
                err2 += abs(s - 1.0)
    if err2 > 1e-10:
        # This shouldn't happen, but fall back
        pass
    
    # Now build 4x4x4 from 7 copies of the 2x2 decomposition
    # The 4x4 matrix is divided into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # Block A11 occupies rows 0-1, cols 0-1
    # Block A12 occupies rows 0-1, cols 2-3
    # Block A21 occupies rows 2-3, cols 0-1
    # Block A22 occupies rows 2-3, cols 2-3
    #
    # For the 4x4 tensor, T4[i,k,j] = 1 for all i,k,j in {0,1,2,3}
    # This can be written as: T4[i,k,j] = sum over block contributions
    #
    # Using Strassen at the block level:
    # Let A_blocks = [[A11, A12], [A21, A22]] (each 2x2)
    # B_blocks = [[B11, B12], [B21, B22]]
    # C_blocks = [[C11, C12], [C21, C22]]
    #
    # C_blocks = A_blocks @ B_blocks (block multiplication)
    # Using Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M2 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    #
    # Each Mx is a 2x2x2x2 matmul, which has rank 7.
    # So total rank = 7 * 7 = 49.
    #
    # For each outer term t and inner term k:
    # The rank-1 term involves:
    #   A elements: linear combo from the "left" side of M_t
    #   B elements: linear combo from the "right" side of M_t
    #   C elements: contribution to C from M_t's k-th inner term
    
    # Define the 7 outer terms with their A and B block coefficients
    # and C block contributions
    
    # For each outer term t:
    #   left_blocks: dict (block_pos) -> coeff, where block_pos is (br,bc) in {0,1}x{0,1}
    #     These are the coefficients for the A blocks in the left factor
    #   right_blocks: dict (block_pos) -> coeff
    #   c_contrib: dict (block_pos) -> coeff, for the C blocks
    
    # M1: left = {A11:1, A22:1}, right = {B11:1, B22:1}
    #   C11 += M1, C22 += M1
    outer_def = [
        # (left_coeffs, right_coeffs, c_coeffs)
        # left_coeffs: {A_block_pos: coeff}
        # right_coeffs: {B_block_pos: coeff}
        # c_coeffs: {C_block_pos: coeff} (multiplied by the full 2x2 product)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For 4x4: A index = i*4+k, B index = k*4+j, C index = j*4+i
    # Block (br,bc) of A occupies rows 2br to 2br+1, cols 2bc to 2bc+1
    # So A element (i,k) is in block (i//2, k//2) at local position (i%2, k%2)
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    for t in range(7):
        left_c, right_c, c_c = outer_def[t]
        for k in range(7):
            li = t * 7 + k
            
            if time.time() - t0 > deadline:
                break
            
            # Get the k-th inner Strassen term
            # U2 has shape (4,7), V2 has shape (4,7), W2 has shape (4,7)
            # U2 index: local_i*2 + local_k
            # V2 index: local_k*2 + local_j
            # W2 index: local_j*2 + local_i
            
            # For the left side:
            # The "2x2 matrix" of A-blocks has entries S_L[br,bc] = left_c.get((br,bc), 0) * A_block(br,bc)
            # The inner Strassen's left 2x2 matrix has entries:
            #   L[0,0] = S_L[0,0] (a 2x2 matrix)
            #   L[0,1] = S_L[0,1]
            #   L[1,0] = S_L[1,0]
            #   L[1,1] = S_L[1,1]
            #
            # The k-th inner term uses l_coeffs from U2:
            # U2[local_i*2+local_k, k] gives the coefficient for L[local_i, local_k]
            #
            # So the coefficient for A element (i, ak) in U is:
            #   sum over (local_i, local_k) such that:
            #     i = 2*br + local_i for some block (br, bc)
            #     ak = 2*bc + local_k
            #   of: U2[local_i*2+local_k, k] * left_c.get((br, bc), 0)
            #
            # Similarly for V and W.
            
            # Build U
            for i in range(4):
                for ak in range(4):
                    br = i // 2
                    bc = ak // 2
                    li_local = i % 2
                    lk_local = ak % 2
                    coeff = U2[li_local*2 + lk_local, k] * left_c.get((br, bc), 0)
                    if coeff != 0:
                        U[i*4+ak, li] = coeff
            
            # Build V
            for bk in range(4):
                for j in range(4):
                    br = bk // 2
                    bc = j // 2
                    lk_local = bk % 2
                    lj_local = j % 2
                    coeff = V2[lk_local*2 + lj_local, k] * right_c.get((br, bc), 0)
                    if coeff != 0:
                        V[bk*4+j, li] = coeff
            
            # Build W
            # W index = j*4+i (C col j, C row i)
            # The C block at position (cbr, cbc) receives contribution from the k-th inner term
            # W2[local_j*2+local_i, k] gives the coefficient for P[local_j, local_i]
            # C block (cbr, cbc) gets c_c.get((cbr, cbc), 0) * P
            # C element (ci, cj) is in block (ci//2, cj//2) at local pos (ci%2, cj%2)
            for cj in range(4):
                for ci in range(4):
                    cbr = ci // 2
                    cbc = cj // 2
                    li_local = ci % 2
                    lj_local = cj % 2
                    coeff = W2[lj_local*2 + li_local, k] * c_c.get((cbr, cbc), 0)
                    if coeff != 0:
                        W[cj*4+ci, li] = coeff
    
    # Verify
    err = 0.0
    for i in range(4):
        for k in range(4):
            for j in range(4):
                s = 0.0
                for r in range(49):
                    s += U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
                err += abs(s - 1.0)
    
    if err > 1e-6:
        # Fallback to schoolbook rank-64
        U2 = np.zeros((16, 64), dtype=np.float64)
        V2 = np.zeros((16, 64), dtype=np.float64)
        W2 = np.zeros((16, 64), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U2[i*4+k, idx] = 1.0
                    V2[k*4+j, idx] = 1.0
                    W2[j*4+i, idx] = 1.0
                    idx += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
