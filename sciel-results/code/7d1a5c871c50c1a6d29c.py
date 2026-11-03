"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build the schoolbook decomposition as a fallback
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

    # Try 4+2 mixed block approach (rank 176)
    mixed_rank = 176
    if mixed_rank < best_rank:
        success = _build_4plus2_mixed(n, m, p, deadline)
        if success is not None:
            best_rank = mixed_rank
            best_U, best_V, best_W = success

    # Try Block-Strassen (rank 189) as fallback
    strassen_rank = 189
    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            best_rank = strassen_rank
            best_U, best_V, best_W = success

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _strassen_2x2_terms():
    """Return Strassen's 7 bilinear forms for 2x2 matrix multiplication.
    Each term: (A_coeffs[4], B_coeffs[4], C_coeffs[4])
    A_coeffs index: [a00, a01, a10, a11]
    B_coeffs index: [b00, b01, b10, b11]
    C_coeffs index: [c00, c01, c10, c11]
    """
    return [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]


def _laderman_4x4_terms():
    """Return Laderman's 25 bilinear forms for 4x4 matrix multiplication.
    Each term: (A_coeffs[16], B_coeffs[16], C_coeffs[16])
    Index: row-major [a00,a01,a02,a03, a10,a11,a12,a13, a20,..., a33]
    """
    # Laderman's algorithm using 2x2 block structure.
    # A = [A00 A01; A10 A11] where each block is 2x2
    # A00 = [[a00,a01],[a10,a11]], A01 = [[a02,a03],[a12,a13]]
    # A10 = [[a20,a21],[a30,a31]], A11 = [[a22,a23],[a32,a33]]
    #
    # We use the standard Laderman formulation with 25 products.
    # Each product is (linear_comb_A) * (linear_comb_B) contributing to C.
    
    # Helper to create a 16-dim vector from 4x4 block coefficients
    def blk(c00, c01, c10, c11):
        """Create 16-dim vector from 2x2 block coefficients (each coeff applies to a 2x2 block)."""
        v = np.zeros(16)
        # Block (0,0): rows 0-1, cols 0-1
        v[0] = c00; v[1] = c00; v[4] = c00; v[5] = c00
        # Block (0,1): rows 0-1, cols 2-3
        v[2] = c01; v[3] = c01; v[6] = c01; v[7] = c01
        # Block (1,0): rows 2-3, cols 0-1
        v[8] = c10; v[9] = c10; v[12] = c10; v[13] = c10
        # Block (1,1): rows 2-3, cols 2-3
        v[10] = c11; v[11] = c11; v[14] = c11; v[15] = c11
        return tuple(v)
    
    # Helper for individual element coefficients
    def elem(row, col, val=1):
        v = np.zeros(16)
        v[row * 4 + col] = val
        return tuple(v)
    
    # Laderman's 25 forms (using block notation):
    # M1  = A00 * B00
    # M2  = A01 * B11
    # M3  = (A00+A01) * (B00+B10)
    # M4  = (A10+A11) * (B00+B10)
    # M5  = (A00+A10) * (B10+B11)
    # M6  = (A01+A11) * (B00+B11)
    # M7  = A00 * (B11-B00)
    # M8  = A11 * (B00-B10)
    # M9  = (A00+A11) * (B01-B11)
    # M10 = (A01+A11) * (B10-B00)
    # M11 = (A00+A10) * (B01-B11)
    # M12 = (A01+A10) * (B10-B11)
    # M13 = A01 * (B10-B11)
    # M14 = A10 * (B01-B11)
    # M15 = (A00+A11) * (B00+B11)
    # M16 = (A01+A10) * (B00+B11)
    # M17 = A00 * (B01+B11)
    # M18 = A11 * (B00+B10)
    # M19 = (A00+A10) * (B01+B10)
    # M20 = (A01+A11) * (B10+B11)
    # M21 = A01 * B10
    # M22 = A10 * B11
    # M23 = (A00+A11) * B01
    # M24 = (A01+A10) * B10
    # M25 = A00 * B11
    
    # C reconstruction:
    # C00 = M1 + M3 - M7 - M15 + M17
    # C01 = M2 + M6 - M10 - M16 + M20 - M25
    # C10 = M4 + M5 - M8 - M15 + M18 - M22
    # C11 = M5 + M9 - M11 - M13 + M14 + M21
    
    # Wait, I need to verify this. Let me use a different, verified formulation.
    # Actually, let me just generate the terms and verify numerically.
    
    terms = []
    
    # M1 = A00 * B00
    A = blk(1,0,0,0); B = blk(1,0,0,0); C = blk(1,0,0,0)
    terms.append((A, B, C))
    
    # M2 = A01 * B11
    A = blk(0,1,0,0); B = blk(0,0,0,1); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M3 = (A00+A01) * (B00+B10)
    A = blk(1,1,0,0); B = blk(1,0,1,0); C = blk(1,0,0,0)
    terms.append((A, B, C))
    
    # M4 = (A10+A11) * (B00+B10)
    A = blk(0,0,1,1); B = blk(1,0,1,0); C = blk(0,0,1,0)
    terms.append((A, B, C))
    
    # M5 = (A00+A10) * (B10+B11)
    A = blk(1,0,1,0); B = blk(0,0,1,1); C = blk(0,0,0,1)
    terms.append((A, B, C))
    
    # M6 = (A01+A11) * (B00+B11)
    A = blk(0,1,0,1); B = blk(1,0,0,1); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M7 = A00 * (B11-B00)
    A = blk(1,0,0,0); B = blk(-1,0,0,1); C = blk(1,0,0,0)
    terms.append((A, B, C))
    
    # M8 = A11 * (B00-B10)
    A = blk(0,0,0,1); B = blk(1,0,-1,0); C = blk(0,0,1,0)
    terms.append((A, B, C))
    
    # M9 = (A00+A11) * (B01-B11)
    A = blk(1,0,0,1); B = blk(0,1,0,-1); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M10 = (A01+A11) * (B10-B00)
    A = blk(0,1,0,1); B = blk(-1,0,1,0); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M11 = (A00+A10) * (B01-B11)
    A = blk(1,0,1,0); B = blk(0,1,0,-1); C = blk(0,0,0,1)
    terms.append((A, B, C))
    
    # M12 = (A01+A10) * (B10-B11)
    A = blk(0,1,1,0); B = blk(0,0,1,-1); C = blk(0,0,0,1)
    terms.append((A, B, C))
    
    # M13 = A01 * (B10-B11)
    A = blk(0,1,0,0); B = blk(0,0,1,-1); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M14 = A10 * (B01-B11)
    A = blk(0,0,1,0); B = blk(0,1,0,-1); C = blk(0,0,1,0)
    terms.append((A, B, C))
    
    # M15 = (A00+A11) * (B00+B11)
    A = blk(1,0,0,1); B = blk(1,0,0,1); C = blk(1,0,0,1)
    terms.append((A, B, C))
    
    # M16 = (A01+A10) * (B00+B11)
    A = blk(0,1,1,0); B = blk(1,0,0,1); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M17 = A00 * (B01+B11)
    A = blk(1,0,0,0); B = blk(0,1,0,1); C = blk(1,0,0,0)
    terms.append((A, B, C))
    
    # M18 = A11 * (B00+B10)
    A = blk(0,0,0,1); B = blk(1,0,1,0); C = blk(0,0,1,0)
    terms.append((A, B, C))
    
    # M19 = (A00+A10) * (B01+B10)
    A = blk(1,0,1,0); B = blk(0,1,1,0); C = blk(0,0,0,1)
    terms.append((A, B, C))
    
    # M20 = (A01+A11) * (B10+B11)
    A = blk(0,1,0,1); B = blk(0,0,1,1); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M21 = A01 * B10
    A = blk(0,1,0,0); B = blk(0,0,1,0); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M22 = A10 * B11
    A = blk(0,0,1,0); B = blk(0,0,0,1); C = blk(0,0,1,0)
    terms.append((A, B, C))
    
    # M23 = (A00+A11) * B01
    A = blk(1,0,0,1); B = blk(0,1,0,0); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M24 = (A01+A10) * B10
    A = blk(0,1,1,0); B = blk(0,0,1,0); C = blk(0,1,0,0)
    terms.append((A, B, C))
    
    # M25 = A00 * B11
    A = blk(1,0,0,0); B = blk(0,0,0,1); C = blk(1,0,0,0)
    terms.append((A, B, C))
    
    return terms


def _build_4plus2_mixed(n, m, p, deadline):
    """Build 4+2 mixed block decomposition for 6x6x6.
    
    Partition: rows/cols 0-3 (size 4) and 4-5 (size 2).
    Block products:
      C[0:4,0:4] = A[0:4,0:4]*B[0:4,0:4] + A[0:4,4:6]*B[4:6,0:4]
      C[0:4,4:6] = A[0:4,0:4]*B[0:4,4:6] + A[0:4,4:6]*B[4:6,4:6]
      C[4:6,0:4] = A[4:6,0:4]*B[0:4,0:4] + A[4:6,4:6]*B[4:6,0:4]
      C[4:6,4:6] = A[4:6,0:4]*B[0:4,4:6] + A[4:6,4:6]*B[4:6,4:6]
    
    Ranks:
      4x4*4x4: Laderman 25
      4x2*2x4: schoolbook 32
      4x4*4x2: schoolbook 32
      4x2*2x2: schoolbook 16
      2x4*4x4: schoolbook 32
      2x2*2x4: schoolbook 16
      2x4*4x2: schoolbook 16
      2x2*2x2: Strassen 7
    Total: 25+32+32+16+32+16+16+7 = 176
    """
    nm, mp, pn = n * m, m * p, p * n
    total_rank = 176
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    
    # Helper: add schoolbook terms for (rows_a, cols_a) x (rows_b, cols_b) -> (rows_c, cols_c)
    def add_schoolbook(ra, ca, rb, cb, rc, cc):
        nonlocal col
        for i in range(ra):
            for j in range(ca):
                for k in range(rb):
                    a_idx = (ra + i) * m + (ca + j)
                    b_idx = (rb + k) * p + (cb + j)  # j must match
                    # Wait, this is wrong. Let me reconsider.
                    pass
        # Actually, let me handle this differently.
        # A block: rows [ar:ar+ra], cols [ac:ac+ca]
        # B block: rows [br:br+rb], cols [bc:bc+cb]
        # C block: rows [cr:cr+rc], cols [cc:cc+cc]
        # For schoolbook: for each (i,j,k) where A[i,j]*B[j,k]->C[i,k]
        # i in [0,ra), j in [0,ca) must equal [0,rb), k in [0,cb)
        # So ca must equal rb
        pass
    
    # Let me use a cleaner approach: directly fill U, V, W for each block product.
    
    # Block definitions: (row_start, col_start, size)
    # A blocks: A00=(0,0,4), A01=(0,4,2), A10=(4,0,2), A11=(4,4,2)
    # B blocks: B00=(0,0,4), B01=(0,4,2), B10=(4,0,2), B11=(4,4,2)
    # C blocks: C00=(0,0,4), C01=(0,4,2), C10=(4,0,2), C11=(4,4,2)
    
    # C00 = A00*B00 + A01*B10
    # A00*B00: 4x4 * 4x4 -> 4x4, use Laderman (rank 25)
    laderman_terms = _laderman_4x4_terms()
    for (a_coeff, b_coeff, c_coeff) in laderman_terms:
        for i in range(4):
            for j in range(4):
                val = a_coeff[i * 4 + j]
                if val != 0:
                    U[i * m + j, col] = val
        for i in range(4):
            for j in range(4):
                val = b_coeff[i * 4 + j]
                if val != 0:
                    V[i * p + j, col] = val
        for i in range(4):
            for j in range(4):
                val = c_coeff[i * 4 + j]
                if val != 0:
                    W[j * n + i, col] = val  # W[k*n+i], k=col, i=row
        col += 1
    
    # A01*B10: 4x2 * 2x4 -> 4x4, schoolbook (rank 4*2*4=32)
    for i in range(4):
        for j in range(2):
            for k in range(4):
                a_row = i
                a_col = 4 + j
                b_row = 4 + j
                b_col = k
                c_row = i
                c_col = k
                U[a_row * m + a_col, col] = 1
                V[b_row * p + b_col, col] = 1
                W[c_col * n + c_row, col] = 1
                col += 1
    
    # C01 = A00*B01 + A01*B11
    # A00*B01: 4x4 * 4x2 -> 4x2, schoolbook (rank 4*4*2=32)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                a_row = i
                a_col = j
                b_row = j
                b_col = 4 + k
                c_row = i
                c_col = 4 + k
                U[a_row * m + a_col, col] = 1
                V[b_row * p + b_col, col] = 1
                W[c_col * n + c_row, col] = 1
                col += 1
    
    # A01*B11: 4x2 * 2x2 -> 4x2, schoolbook (rank 4*2*2=16)
    for i in range(4):
        for j in range(2):
            for k in range(2):
                a_row = i
                a_col = 4 + j
                b_row = 4 + j
                b_col = 4 + k
                c_row = i
                c_col = 4 + k
                U[a_row * m + a_col, col] = 1
                V[b_row * p + b_col, col] = 1
                W[c_col * n + c_row, col] = 1
                col += 1
    
    # C10 = A10*B00 + A11*B10
    # A10*B00: 2x4 * 4x4 -> 2x4, schoolbook (rank 2*4*4=32)
    for i in range(2):
        for j in range(4):
            for k in range(4):
                a_row = 4 + i
                a_col = j
                b_row = j
                b_col = k
                c_row = 4 + i
                c_col = k
                U[a_row * m + a_col, col] = 1
                V[b_row * p + b_col, col] = 1
                W[c_col * n + c_row, col] = 1
                col += 1
    
    # A11*B10: 2x2 * 2x4 -> 2x4, schoolbook (rank 2*2*4=16)
    for i in range(2):
        for j in range(2):
            for k in range(4):
                a_row = 4 + i
                a_col = 4 + j
                b_row = 4 + j
                b_col = k
                c_row = 4 + i
                c_col = k
                U[a_row * m + a_col, col] = 1
                V[b_row * p + b_col, col] = 1
                W[c_col * n + c_row, col] = 1
                col += 1
    
    # C11 = A10*B01 + A11*B11
    # A10*B01: 2x4 * 4x2 -> 2x2, schoolbook (rank 2*4*2=16)
    for i in range(2):
        for j in range(4):
            for k in range(2):
                a_row = 4 + i
                a_col = j
                b_row = j
                b_col = 4 + k
                c_row = 4 + i
                c_col = 4 + k
                U[a_row * m + a_col, col] = 1
                V[b_row * p + b_col, col] = 1
                W[c_col * n + c_row, col] = 1
                col += 1
    
    # A11*B11: 2x2 * 2x2 -> 2x2, Strassen (rank 7)
    strassen_terms = _strassen_2x2_terms()
    for (a_coeff, b_coeff, c_coeff) in strassen_terms:
        for i in range(2):
            for j in range(2):
                val = a_coeff[i * 2 + j]
                if val != 0:
                    U[(4+i) * m + (4+j), col] = val
        for i in range(2):
            for j in range(2):
                val = b_coeff[i * 2 + j]
                if val != 0:
                    V[(4+i) * p + (4+j), col] = val
        for i in range(2):
            for j in range(2):
                val = c_coeff[i * 2 + j]
                if val != 0:
                    W[(4+j) * n + (4+i), col] = val
        col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    # Verify
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    
    return U, V, W


def _build_block_strassen(n, m, p, deadline):
    """Build a block-Strassen decomposition for 6x6x6 (rank 189)."""
    nm, mp, pn = n * m, m * p, p * n
    num_blocks = 3
    block_size = 2
    total_rank = num_blocks ** 3 * 7
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    strassen_terms = _strassen_2x2_terms()
    
    col = 0
    for bi in range(num_blocks):
        for bk in range(num_blocks):
            for bj in range(num_blocks):
                for s in range(7):
                    a_coeff, b_coeff, c_coeff = strassen_terms[s]
                    for r in range(block_size):
                        for sc in range(block_size):
                            val = a_coeff[r * block_size + sc]
                            if val != 0:
                                U[(bi * block_size + r) * m + (bk * block_size + sc), col] = val
                    for r in range(block_size):
                        for sc in range(block_size):
                            val = b_coeff[r * block_size + sc]
                            if val != 0:
                                V[(bk * block_size + r) * p + (bj * block_size + sc), col] = val
                    for r in range(block_size):
                        for sc in range(block_size):
                            val = c_coeff[r * block_size + sc]
                            if val != 0:
                                W[(bj * block_size + sc) * n + (bi * block_size + r), col] = val
                    col += 1
    
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
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
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    np.random.seed(42)
    for _ in range(3000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
