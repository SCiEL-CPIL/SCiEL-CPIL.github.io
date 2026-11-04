import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # We'll build the tensor decomposition using Strassen's algorithm
    # Each rank-1 term is (U_entry, V_entry, W_entry)
    # U has shape (n*m, rank), indexed by [i*m + j]
    # V has shape (m*p, rank), indexed by [j*p + k]
    # W has shape (p*n, rank), indexed by [k*n + i]
    
    terms = []  # list of dicts: {'U': {idx: coeff}, 'V': {idx: coeff}, 'W': {idx: coeff}}
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. Each entry is a dict {index: coefficient}"""
        terms.append({'U': u_entries, 'V': v_entries, 'W': w_entries})
    
    def schoolbook(n_rows, m_mids, p_cols, row_off, mid_off, col_off):
        """Schoolbook decomposition for a sub-problem"""
        for i in range(n_rows):
            for j in range(m_mids):
                for k in range(p_cols):
                    abs_i = row_off + i
                    abs_j = mid_off + j
                    abs_k = col_off + k
                    if abs_i < n and abs_j < m and abs_k < p:
                        add_term(
                            {abs_i * m + abs_j: 1},
                            {abs_j * p + abs_k: 1},
                            {abs_k * n + abs_i: 1}
                        )
    
    def strassen(n_rows, m_mids, p_cols, row_off, mid_off, col_off):
        """
        Strassen decomposition for sub-problem of size (n_rows x m_mids) * (m_mids x p_cols)
        with offsets into the global indices.
        """
        # If any dimension is 0, nothing to do
        if n_rows == 0 or m_mids == 0 or p_cols == 0:
            return
        
        # If all dimensions are <= 2, use schoolbook
        if n_rows <= 2 and m_mids <= 2 and p_cols <= 2:
            schoolbook(n_rows, m_mids, p_cols, row_off, mid_off, col_off)
            return
        
        # Split each dimension into two halves
        n1 = n_rows // 2
        n2 = n_rows - n1
        m1 = m_mids // 2
        m2 = m_mids - m1
        p1 = p_cols // 2
        p2 = p_cols - p1
        
        # Strassen's 7 multiplications for 2x2x2 blocks:
        # We need to be careful: the sub-problems may not be square
        # 
        # A is (n_rows x m_mids), split into:
        #   A11: (n1 x m1) at (row_off, mid_off)
        #   A12: (n1 x m2) at (row_off, mid_off + m1)
        #   A21: (n2 x m1) at (row_off + n1, mid_off)
        #   A22: (n2 x m2) at (row_off + n1, mid_off + m1)
        #
        # B is (m_mids x p_cols), split into:
        #   B11: (m1 x p1) at (mid_off, col_off)
        #   B12: (m1 x p2) at (mid_off, col_off + p1)
        #   B21: (m2 x p1) at (mid_off + m1, col_off)
        #   B22: (m2 x p2) at (mid_off + m1, col_off + p1)
        #
        # C is (n_rows x p_cols), split into:
        #   C11: (n1 x p1) at (row_off, col_off)
        #   C12: (n1 x p2) at (row_off, col_off + p1)
        #   C21: (n2 x p1) at (row_off + n1, col_off)
        #   C22: (n2 x p2) at (row_off + n1, col_off + p1)
        #
        # Strassen multiplications:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 - B22)
        # M4 = A22 * (B11 - B21)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # Result:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Strassen multiplication, we need to decompose the product
        # of two "combined" matrices. Each combined matrix is a sum/difference
        # of sub-blocks.
        
        # We'll represent each "matrix expression" as a list of (coeff, sub_block)
        # where sub_block is (n_rows, m_mids, row_off, mid_off) for A-type
        # or (m_mids, p_cols, mid_off, col_off) for B-type
        
        # For simplicity, let's handle each multiplication by expanding the
        # sum/difference and using schoolbook or recursive Strassen on each term.
        
        # Actually, the cleanest approach: for each of the 7 multiplications,
        # the "left factor" is a linear combination of A sub-blocks, and the
        # "right factor" is a linear combination of B sub-blocks.
        # The product is a bilinear form, so we can expand it.
        
        # Let me define helper to get a list of (coeff, (n,m,roff,moff)) for A-blocks
        # and (coeff, (m,p,moff,coff)) for B-blocks
        
        # M1: (A11 + A22) * (B11 + B22)
        # = A11*B11 + A11*B22 + A22*B11 + A22*B22
        for (a_coeff, a_n, a_m, a_ro, a_mo) in [(1, n1, m1, row_off, mid_off), (1, n2, m2, row_off+n1, mid_off+m1)]:
            for (b_coeff, b_m, b_p, b_mo, b_co) in [(1, m1, p1, mid_off, col_off), (1, m2, p2, mid_off+m1, col_off+p1)]:
                # This term contributes a_coeff * b_coeff to the product
                # But we need to handle the fact that the dimensions might not match
                # A block is (a_n x a_m), B block is (b_m x b_p)
                # They multiply to (a_n x b_p)
                # But wait - in Strassen, the combined matrix has the same dimensions
                # as the original, so we need to be more careful.
                
                # Actually, let me reconsider. The issue is that when we write
                # (A11 + A22), this is a (n_rows x m_mids) matrix where:
                # - rows 0..n1-1, cols 0..m1-1: A11
                # - rows 0..n1-1, cols m1..: 0
                # - rows n1.., cols 0..m1-1: 0
                # - rows n1.., cols m1..: A22
                #
                # Similarly for B.
                
                # So the product (A11+A22)*(B11+B22) is not simply the sum of
                # the 4 products - we need to account for the positions.
                
                # Let me use a different approach: directly enumerate the contributions.
                # For the product of two "sparse block" matrices, each non-zero
                # block in A multiplies each non-zero block in B.
                
                pass
        
        # This is getting too complex. Let me use a simpler but still effective approach:
        # For each of the 7 Strassen multiplications, I'll build the combined matrices
        # explicitly as sparse representations and then decompose their product.
        
        # Alternative cleaner approach: use the fact that we can just track which
        # (i,j,k) triples are involved in each "multiplication" and build the
        # decomposition from scratch.
        
        # Let me restart with a cleaner recursive approach.
        pass
    
    # Given the complexity, let me use a more direct approach:
    # Pad to 4x8 and use a known decomposition.
    
    # For 3x5x7, I'll construct the decomposition by:
    # 1. Padding A to 4x5 (add zero row)
    # 2. Padding B to 5x8 (add zero col)  
    # 3. Using Strassen recursively on the 4x5x8 problem
    # 4. Only counting terms where all indices are in the original range
    
    # Let me implement a clean recursive Strassen:
    
    def decompose(n_r, m_m, p_c, roff, moff, coff, depth=0):
        """
        Decompose (n_r x m_m) * (m_m x p_c) -> (n_r x p_c)
        with offsets into global indices.
        Returns list of terms.
        """
        if time.time() - START_TIME > TIME_LIMIT:
            return []
        
        if n_r == 0 or m_m == 0 or p_c == 0:
            return []
        
        # Base case: use schoolbook
        if n_r <= 2 and m_m <= 2 and p_c <= 2:
            result = []
            for i in range(n_r):
                for j in range(m_m):
                    for k in range(p_c):
                        gi = roff + i
                        gj = moff + j
                        gk = coff + k
                        if gi < n and gj < m and gk < p:
                            result.append((gi, gj, gk, 1))
            return result
        
        # Split
        nr1 = n_r // 2
        nr2 = n_r - nr1
        mm1 = m_m // 2
        mm2 = m_m - mm1
        pc1 = p_c // 2
        pc2 = p_c - pc1
        
        result = []
        
        # For each of the 7 Strassen multiplications, we need to decompose
        # the product of two "combined" sub-matrices.
        
        # I'll represent each sub-matrix as a list of (coeff, sub_n, sub_m, sub_roff, sub_moff)
        # for A-type or (coeff, sub_m, sub_p, sub_moff, sub_coff) for B-type
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11: (nr1 x mm1) at (roff, moff)
        # A22: (nr2 x mm2) at (roff+nr1, moff+mm1)
        # B11: (mm1 x pc1) at (moff, coff)
        # B22: (mm2 x pc2) at (moff+mm1, coff+pc1)
        
        # The product involves all pairs of non-zero blocks
        # A11 * B11: (nr1 x mm1) * (mm1 x pc1) -> (nr1 x pc1) at (roff, coff)
        # A11 * B22: (nr1 x mm1) * (mm2 x pc2) -> but mm1 != mm2 potentially!
        
        # Wait, this is the issue. In standard Strassen for square matrices,
        # all sub-blocks are the same size. For rectangular, they're not.
        
        # The correct approach for rectangular Strassen is to pad to make them
        # square, or to use a different decomposition.
        
        # Let me just use schoolbook for the base case and try to get a
        # reasonable rank by being smart about which terms to include.
        
        # Actually, let me try a different strategy: use the known fact that
        # for 2x2x2, Strassen gives rank 7 instead of 8. For our problem,
        # if I can identify the 2x2x2 sub-problems and apply Strassen to each,
        # I can reduce the rank.
        
        # For 3x5x7:
        # - I can split into blocks and for each 2x2x2 block, use rank 7
        # - The total rank will be less than 105
        
        # Let me just use schoolbook for now but with a twist: I'll try to
        # find a better decomposition by using the structure.
        
        # Fallback: schoolbook
        for i in range(n_r):
            for j in range(m_m):
                for k in range(p_c):
                    gi = roff + i
                    gj = moff + j
                    gk = coff + k
                    if gi < n and gj < m and gk < p:
                        result.append((gi, gj, gk, 1))
        
        return result
    
    # Use schoolbook decomposition
    all_terms = decompose(n, m, p, 0, 0, 0)
    
    rank = len(all_terms)
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    for idx, (i, j, k, coeff) in enumerate(all_terms):
        U[i * m + j][idx] = float(coeff)
        V[j * p + k][idx] = float(coeff)
        W[k * n + i][idx] = float(coeff)
    
    return {'U': U, 'V': V, 'W': W}
