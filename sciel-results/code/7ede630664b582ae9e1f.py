"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _laderman_4x4():
    """
    Laderman's rank-17 algorithm for 4x4 matrix multiplication.
    Returns U, V, W as lists of tuples.
    U: (i,j) pairs for A
    V: (j,k) pairs for B
    W: (k,i) pairs for C
    """
    # Laderman's 17 terms for 4x4 matrix multiplication
    # A is 4x4, B is 4x4, C = A*B
    # Indices 0..3 for rows/cols
    
    # The 17 bilinear forms:
    # We use the standard Laderman decomposition.
    # Let A[i,j], B[j,k]
    
    # Term 1: A[0,0]*B[0,0] -> C[0,0]
    # Term 2: A[0,0]*B[0,1] + A[0,1]*B[0,0] ... etc.
    
    # Actually, let me use the well-known explicit Laderman algorithm.
    # For 4x4:
    # P1 = A[0,0] * B[0,0]
    # P2 = A[0,0] * B[0,1] + A[0,1] * B[0,0]
    # ... this is getting complex. Let me use a verified formulation.
    
    # Standard Laderman 4x4 decomposition (17 multiplications):
    # Using indices 0,1,2,3
    
    U_terms = []  # (i, j)
    V_terms = []  # (j, k)
    W_terms = []  # (k, i, sign) - actually W maps to C[k,i] position
    
    # Let me use the formulation where each term is:
    # P_t = sum_{(i,j) in S_t} A[i,j] * sum_{(j,k) in T_t} B[j,k]
    # C[k,i] += sign * P_t
    
    # Laderman's algorithm for 4x4:
    # The 17 terms are defined as follows (using 0-indexed):
    
    # I'll use a known correct set of 17 bilinear forms.
    # From Laderman (1976), the 4x4 algorithm:
    
    terms = [
        # (A_indices, B_indices, C_indices_with_signs)
        # Each term: list of (i,j) for A, list of (j,k) for B, list of (k,i,coeff) for C
        
        # T1
        ([(0,0)], [(0,0)], [(0,0,1)]),
        # T2
        ([(0,0)], [(0,1)], [(0,1,1)]),
        # T3
        ([(0,1)], [(1,0)], [(1,0,1)]),
        # T4
        ([(1,1)], [(1,1)], [(1,1,1)]),
        # T5
        ([(0,0),(0,1)], [(1,0),(0,0)], [(1,0,1),(0,0,-1)]),
        # T6
        ([(0,0),(0,1)], [(1,1),(0,1)], [(1,1,1),(0,1,-1)]),
        # T7
        ([(2,0)], [(0,2)], [(2,0,1)]),
        # T8
        ([(2,0)], [(0,3)], [(3,0,1)]),
        # T9
        ([(2,1)], [(1,2)], [(2,2,1)]),
        # T10
        ([(2,1)], [(1,3)], [(3,2,1)]),
        # T11
        ([(3,0)], [(0,2)], [(2,1,1)]),
        # T12
        ([(3,0)], [(0,3)], [(3,1,1)]),
        # T13
        ([(3,1)], [(1,2)], [(2,3,1)]),
        # T14
        ([(3,1)], [(1,3)], [(3,3,1)]),
        # T15
        ([(2,0),(2,1)], [(0,2),(1,2)], [(2,2,1),(2,0,-1)]),
        # T16
        ([(2,0),(2,1)], [(0,3),(1,3)], [(3,2,1),(2,1,-1)]),
        # T17
        ([(3,0),(3,1)], [(0,2),(1,2)], [(2,3,1),(2,1,-1)]),
    ]
    
    # This is getting error-prone. Let me use a different, more reliable approach.
    # I'll construct the Laderman algorithm programmatically from its known structure.
    
    # Actually, for a 6x6 problem, let me think about what's most practical.
    # 
    # The known best rank for 6x6 matrix multiplication:
    # - Schoolbook: 216
    # - Block Strassen with 2x2 blocks of 3x3: uses Strassen on 2x2 block matrix
    #   where each block is 3x3. Strassen for 2x2 uses 7 multiplications of 3x3.
    #   Each 3x3 multiplication can be done with schoolbook (27) or better.
    #   Total: 7 * 27 = 189 if we use schoolbook for the 3x3 blocks.
    #   
    # - Alternative: Partition 6x6 into 3x3 grid of 2x2 blocks.
    #   Use Strassen on the 3x3 block matrix? No, Strassen is for 2x2.
    #   
    # Let me go with Block Strassen: 2x2 blocks of 3x3.
    # Strassen's 2x2 algorithm:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1+M4-M5+M7
    #   C12 = M3+M5
    #   C21 = M2+M4
    #   C22 = M1-M2+M3+M6
    #
    # Each Mi is a 3x3 matrix product. If we do each with schoolbook (rank 27),
    # total rank = 7 * 27 = 189.
    
    return None  # placeholder, we'll do block Strassen instead


def _block_strassen(n, m, p, block_size):
    """
    Block Strassen: partition n x m and m x p into 2x2 blocks of size block_size.
    Requires n = 2*block_size, m = 2*block_size, p = 2*block_size.
    Each block multiplication uses schoolbook.
    Total rank = 7 * block_size^3.
    """
    # For 6x6: block_size = 3, so we have 2x2 blocks of 3x3.
    # n=6, m=6, p=6, block_size=3
    bs = block_size
    # 7 Strassen multiplications, each of size bs x bs
    # Each bs x bs schoolbook has rank bs^3
    
    # We need to build U, V, W for the full 6x6 problem.
    # U has shape (n*m, rank) = (36, 7*bs^3)
    # V has shape (m*p, rank) = (36, 7*bs^3)
    # W has shape (p*n, rank) = (36, 7*bs^3)
    
    total_rank = 7 * bs * bs * bs
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block indices:
    # A = [[A11, A12],
    #      [A21, A22]]
    # B = [[B11, B12],
    #      [B21, B22]]
    # C = [[C11, C12],
    #      [C21, C22]]
    #
    # A11: rows 0..bs-1, cols 0..bs-1
    # A12: rows 0..bs-1, cols bs..2bs-1
    # A21: rows bs..2bs-1, cols 0..bs-1
    # A22: rows bs..2bs-1, cols bs..2bs-1
    # Similarly for B and C.
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term Mt, we need to express it as a bilinear form.
    # Mt = (sum of signed A blocks) * (sum of signed B blocks)
    # Each A block is a bs x bs matrix, each B block is a bs x bs matrix.
    # The product is a bs x bs matrix.
    
    # For a single bs x bs schoolbook multiplication:
    # (A_block * B_block)[r, c] = sum_d A_block[r, d] * B_block[d, c]
    # In our indexing:
    # A_block[r, d] corresponds to A[i, j] where i is the row in the full matrix, j is the col.
    # For block A11: i in [0,bs), j in [0,bs)
    # For block A12: i in [0,bs), j in [bs, 2bs)
    # etc.
    
    # Let's define a helper to add a schoolbook multiplication of two blocks.
    
    def block_row_range(block_row):
        """Return (start, end) row indices for block_row (0 or 1)."""
        return block_row * bs, (block_row + 1) * bs
    
    def block_col_range(block_col):
        """Return (start, end) col indices for block_col (0 or 1)."""
        return block_col * bs, (block_col + 1) * bs
    
    def add_schoolbook_term(U, V, W, col_offset, a_blocks, b_blocks, c_blocks):
        """
        Add a schoolbook multiplication term.
        a_blocks: list of (block_row, block_col, sign) for A
        b_blocks: list of (block_row, block_col, sign) for B
        c_blocks: list of (block_row, block_col, sign) for C
        This computes: C += sum_{cbr,cbc,cs} cs * (sum_{abr,abc,as} as * A[abr,abc]) * (sum_{bbr,bbc,bs} bs * B[bbr,bbc])
        where the result goes to C[cbr, cbc].
        
        Actually, this is more complex because Strassen combines blocks before multiplying.
        Let me think differently.
        
        Each Strassen term Mt is: (linear combination of A blocks) * (linear combination of B blocks)
        The linear combinations are bs x bs matrices.
        Then we multiply these two bs x bs matrices using schoolbook.
        Then the result (bs x bs) is added to C blocks with appropriate signs.
        """
        pass
    
    # Let me use a different, more direct approach.
    # I'll enumerate all the elementary bilinear terms.
    
    # For Strassen term Mt:
    # Left factor: L_t = sum_{(br, bc, s)} s * A[br, bc]  (a bs x bs matrix)
    # Right factor: R_t = sum_{(br, bc, s)} s * B[br, bc]  (a bs x bs matrix)
    # Product: P_t = L_t * R_t (bs x bs)
    # C contribution: C[cr, cc] += s * P_t
    
    # For each elementary multiplication in the schoolbook expansion of L_t * R_t:
    # P_t[r, c] = sum_d L_t[r, d] * R_t[d, c]
    # L_t[r, d] = sum_{(abr, abc, s)} s * A[abr*bs + r, abc*bs + d]
    # R_t[d, c] = sum_{(bbr, bbc, s)} s * B[bbr*bs + d, bbc*bs + c]
    
    # So P_t[r, c] = sum_d sum_{(abr,abc,s1)} sum_{(bbr,bbc,s2)} s1*s2 * A[abr*bs+r, abc*bs+d] * B[bbr*bs+d, bbc*bs+c]
    
    # And C[cr*bs+r, cc*bs+c] += s_c * P_t[r,c]
    
    # This is a sum of products of A and B entries, each contributing to a C entry.
    # We can collect all these into the U, V, W matrices.
    
    # Define the 7 Strassen terms:
    strassen_terms = [
        # (left_blocks, right_blocks, c_blocks)
        # Each block spec: (block_row, block_col, sign)
        # M1 = (A11+A22)(B11+B22)
        {
            'left': [(0,0,1), (1,1,1)],
            'right': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]  # C11 += M1, C22 += M1
        },
        # M2 = (A21+A22)B11
        {
            'left': [(1,0,1), (1,1,1)],
            'right': [(0,0,1)],
            'c': [(1,0,1), (1,1,-1)]  # C21 += M2, C22 -= M2
        },
        # M3 = A11(B21-B22)
        {
            'left': [(0,0,1)],
            'right': [(1,0,1), (1,1,-1)],
            'c': [(0,1,1), (1,1,1)]  # C12 += M3, C22 += M3
        },
        # M4 = A22(B12-B11)
        {
            'left': [(1,1,1)],
            'right': [(0,1,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]  # C11 += M4, C21 += M4
        },
        # M5 = (A11+A12)B22
        {
            'left': [(0,0,1), (0,1,1)],
            'right': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)]  # C11 -= M5, C12 += M5
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'left': [(1,0,1), (0,0,-1)],
            'right': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]  # C22 += M6
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'left': [(0,1,1), (1,1,-1)],
            'right': [(1,0,1), (1,1,1)],
            'c': [(0,0,1)]  # C11 += M7
        },
    ]
    
    col = 0
    for term in strassen_terms:
        left_blocks = term['left']
        right_blocks = term['right']
        c_blocks = term['c']
        
        # For each (r, c, d) in the bs x bs x bs space:
        for r in range(bs):
            for c in range(bs):
                for d in range(bs):
                    # L_t[r, d] = sum_{(abr, abc, s)} s * A[abr*bs+r, abc*bs+d]
                    # R_t[d, c] = sum_{(bbr, bbc, s)} s * B[bbr*bs+d, bbc*bs+c]
                    # Product term: A[abr*bs+r, abc*bs+d] * B[bbr*bs+d, bbc*bs+c]
                    # with coefficient s_left * s_right
                    # contributes to C[cr*bs+r, cc*bs+c] with additional sign s_c
                    
                    for (abr, abc, s_l) in left_blocks:
                        for (bbr, bbc, s_r) in right_blocks:
                            for (cr, cc, s_c) in c_blocks:
                                # A index: row = abr*bs + r, col = abc*bs + d
                                a_row = abr * bs + r
                                a_col = abc * bs + d
                                a_idx = a_row * m + a_col  # n*m indexing: i*m+j
                                
                                # B index: row = bbr*bs + d, col = bbc*bs + c
                                b_row = bbr * bs + d
                                b_col = bbc * bs + c
                                b_idx = b_row * p + b_col  # m*p indexing: j*p+k
                                
                                # C index: row = cr*bs + r, col = cc*bs + c
                                c_row = cr * bs + r
                                c_col = cc * bs + c
                                c_idx = c_row * n + c_col  # p*n indexing: k*n+i -> wait
                                
                                # W[k*n+i, r] convention: C[k, i] is at position k*n + i
                                # Here c_row is k (row of C), c_col is i (col of C)
                                # So c_idx = c_row * n + c_col
                                
                                coeff = s_l * s_r * s_c
                                U[a_idx, col] += coeff
                                V[b_idx, col] += 1
                                W[c_idx, col] += 1
                                # Wait, this isn't right. The coefficient should be split between U and V or W.
                                # Actually, in a rank-1 decomposition:
                                # C[k,i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                # No wait, the standard tensor decomposition is:
                                # T_{ijk} = sum_r U_{ij, r} * V_{jk, r} * W_{ki, r}
                                # where T_{ijk} is the coefficient of A[i,j]*B[j,k] in C[k,i].
                                # For matrix multiplication, T_{ijk} = 1 for all i,j,k.
                                # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
                                
                                # In our construction, each "column" r represents a bilinear term.
                                # For a term like A[a_row, a_col] * B[b_row, b_col] contributing to C[c_row, c_col]:
                                # We set U[a_row*m+a_col, r] = coeff, V[b_row*p+b_col, r] = 1, W[c_row*n+c_col, r] = 1
                                # But this only works if the term is a single product.
                                # If there are multiple A entries or B entries in one Strassen sub-term,
                                # we need to handle it differently.
                                
                                # Actually, the way I've set it up, each (r, c, d) loop iteration
                                # with specific block combinations gives a single product of one A entry and one B entry.
                                # So U[a_idx, col] = coeff, V[b_idx, col] = 1, W[c_idx, col] = 1 is correct
                                # IF we create a new column for each such elementary product.
                                
                                # But I'm reusing the same column for all of them, which is wrong.
                                # Let me fix: each elementary product gets its own column.
                                
                                # Actually no - I should increment col for each elementary product.
                                # Let me restructure.
                                
                                pass
                    col += 1
    
    # This approach is getting complicated. Let me just use the simple block Strassen
    # where each of the 7 multiplications is a full schoolbook multiplication of bs x bs.
    # That gives rank = 7 * bs^3.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try Block Strassen: 2x2 blocks of 3x3
    # Rank = 7 * 3^3 = 189
    # This is better than schoolbook rank 216.
    
    bs = 3  # block size
    # Verify dimensions
    if n != 2 * bs or m != 2 * bs or p != 2 * bs:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    total_rank = 7 * bs * bs * bs
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen terms definition:
    # Each term: (left_block_specs, right_block_specs, c_block_specs)
    # Block spec: (block_row, block_col, sign)
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for left_blocks, right_blocks, c_blocks in strassen_terms:
        # For each (r, c, d) in bs x bs x bs:
        for r in range(bs):
            for c in range(bs):
                for d in range(bs):
                    for (abr, abc, s_l) in left_blocks:
                        for (bbr, bbc, s_r) in right_blocks:
                            for (cr, cc, s_c) in c_blocks:
                                a_row = abr * bs + r
                                a_col = abc * bs + d
                                a_idx = a_row * m + a_col
                                
                                b_row = bbr * bs + d
                                b_col = bbc * bs + c
                                b_idx = b_row * p + b_col
                                
                                c_row = cr * bs + r
                                c_col = cc * bs + c
                                c_idx = c_row * n + c_col
                                
                                coeff = s_l * s_r * s_c
                                U[a_idx, col] = coeff
                                V[b_idx, col] = 1
                                W[c_idx, col] = 1
                                col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
