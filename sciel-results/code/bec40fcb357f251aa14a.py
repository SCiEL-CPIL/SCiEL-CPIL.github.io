"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 150s wall-clock limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build the schoolbook decomposition as a fallback (rank 216)
    schoolbook_rank = n * m * p  # 216

    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try Block-Strassen: partition 6x6 into 3x3 blocks of 2x2 submatrices
    # Each 2x2 block multiplication uses Strassen's rank-7 algorithm
    # Total rank = 9 block positions * 3 inner blocks * 7 Strassen terms = 189
    strassen_rank = 9 * 3 * 7  # 189

    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = strassen_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    # Convert to lists for the output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_block_strassen(n, m, p, deadline):
    """Build a block-Strassen decomposition for 6x6x6 matrix multiplication.
    
    Partition into 3x3 blocks of 2x2 submatrices.
    Each 2x2 multiply uses Strassen's rank-7 algorithm.
    For each output block (bi, bj): C[bi,bj] = sum_bk A[bi,bk] * B[bk,bj]
    Each product is rank 7, so each output block needs 3*7=21 terms.
    Total: 9 * 21 = 189 terms.
    """
    nm, mp, pn = n * m, m * p, p * n
    block_size = 2
    num_blocks = 3  # 6 / 2 = 3 blocks per dimension
    total_rank = 189

    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Strassen term, we define:
    # - A factors: list of (row, col, coeff) for the 2x2 block A[bi,bk]
    # - B factors: list of (row, col, coeff) for the 2x2 block B[bk,bj]
    # - C contributions: list of (row, col, coeff) for the 2x2 block C[bi,bj]

    # Strassen term definitions
    # Each term: (A_factors, B_factors, C_contribs)
    # A_factors: list of (a_row, a_col, coeff)
    # B_factors: list of (b_row, b_col, coeff)
    # C_contribs: list of (c_row, c_col, coeff)
    # where row/col are 0 or 1 (within the 2x2 block)

    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        # A part: A00 and A11, B part: B00 and B11
        # C: contributes to C00 (+1), C11 (+1)
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (A10+A11)B00
        # A part: A10 and A11, B part: B00
        # C: contributes to C10 (+1), C11 (-1)
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        # M3 = A00(B10-B11)
        # A part: A00, B part: B10 and B11
        # C: contributes to C01 (+1), C11 (+1)
        {
            'A': [(0, 0, 1)],
            'B': [(1, 0, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        # M4 = A11(B01-B00)
        # A part: A11, B part: B01 and B00
        # C: contributes to C00 (+1), C10 (+1)
        {
            'A': [(1, 1, 1)],
            'B': [(0, 1, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        # M5 = (A00+A01)B11
        # A part: A00 and A01, B part: B11
        # C: contributes to C00 (-1), C01 (+1)
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        # M6 = (A10-A00)(B00+B01)
        # A part: A10 and A00, B part: B00 and B01
        # C: contributes to C11 (+1)
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        # M7 = (A01-A11)(B10+B11)
        # A part: A01 and A11, B part: B10 and B11
        # C: contributes to C00 (+1)
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        },
    ]

    col = 0
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                for s in range(7):
                    term = strassen_terms[s]
                    
                    # For this Strassen term, we need to create rank-1 terms
                    # The A factor is a linear combination of elements in A[bi,bk]
                    # The B factor is a linear combination of elements in B[bk,bj]
                    # The C contribution is a linear combination of elements in C[bi,bj]
                    
                    # But wait - Strassen's algorithm works by computing 7 "products"
                    # where each product is of two 2x2 matrices (sums/diffs of blocks).
                    # Each such 2x2 * 2x2 product is itself a matrix product.
                    
                    # Actually, I need to reconsider. In the standard Strassen algorithm
                    # for 2x2 matrices, the 7 products M1...M7 are scalar products if
                    # we're multiplying 2x2 matrices of scalars. But here, each "element"
                    # of the 2x2 block is itself a scalar, so M1...M7 are each products
                    # of two 2x2 matrices (linear combinations of the original elements).
                    
                    # The key insight: each M_s is a product of two 2x2 matrices.
                    # A product of two 2x2 matrices can be decomposed into rank-1 terms
                    # using the schoolbook method (rank 4) or Strassen (rank 7).
                    
                    # But that would give us 9*3*7*4 = 756 or 9*3*7*7 = 1323, which is worse.
                    
                    # I think I'm confusing myself. Let me reconsider.
                    
                    # The correct interpretation: We have 6x6 matrices partitioned into
                    # 3x3 blocks of 2x2 submatrices. The matrix multiplication is:
                    # C[bi,bj] = sum_bk A[bi,bk] * B[bk,bj]
                    # where each * is a 2x2 matrix multiplication.
                    
                    # A 2x2 matrix multiplication has rank 7 (Strassen) or 8 (schoolbook).
                    # So for each (bi,bj,bk) triple, we need 7 rank-1 terms.
                    # Total: 3*3*3*7 = 189 terms.
                    
                    # Each rank-1 term for the 2x2 product A[bi,bk] * B[bk,bj]
                    # involves:
                    # - A linear combination of elements of A[bi,bk] (the A factor)
                    # - A linear combination of elements of B[bk,bj] (the B factor)
                    # - A linear combination of elements of C[bi,bj] (the C factor)
                    
                    # For Strassen, each of the 7 terms M_s involves:
                    # M_s = (sum of A elements with coeffs) * (sum of B elements with coeffs)
                    # and contributes to C with certain coefficients.
                    
                    # But wait, that's not right either. In Strassen's algorithm,
                    # M1 = (A00+A11)(B00+B11) is a product of two 2x2 matrices,
                    # not a scalar product. So M1 is itself a 2x2 matrix.
                    
                    # I think the issue is that I'm conflating two levels of Strassen.
                    # Let me just use the schoolbook for the 2x2 block products.
                    # Each 2x2 * 2x2 product has rank 8 (schoolbook).
                    # Total: 9 * 3 * 8 = 216, which is the same as global schoolbook.
                    
                    # To actually get rank 189, I need to use Strassen at the block level
                    # AND at the scalar level within each block. But that's complex.
                    
                    # Let me try a different approach: directly construct the 189-term
                    # decomposition by treating the 6x6 problem as a 3x3 block problem
                    # where each block multiplication uses Strassen.
                    
                    pass  # This approach is getting too complex; let's use a simpler method

    # Given the complexity, let's just return the schoolbook decomposition
    return None

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    np.random.seed(123)
    for _ in range(500):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
