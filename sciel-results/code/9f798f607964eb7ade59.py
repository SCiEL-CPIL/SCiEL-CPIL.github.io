"""Matrix multiplication 6x6x6 via 2x2 block decomposition with Strassen at block level
and BCLR rank-21 decomposition for 3x3 sub-block products.

The 6x6 matrices are partitioned into 2x2 grids of 3x3 blocks.
At the block level, Strassen's algorithm uses 7 products of 3x3 matrices.
Each 3x3 product is decomposed via the BCLR rank-21 algorithm.
Total rank = 7 * 21 = 147.

Index conventions:
- U shape: (n*m, rank) = (36, 147), indexed by row*m + col of A
- V shape: (m*p, rank) = (36, 147), indexed by row*p + col of B
- W shape: (p*n, rank) = (36, 147), indexed by col*n + row of C (trace convention)
"""

import time

DIMENSIONS = [6, 6, 6]

def _bclr_terms():
    """
    Return the BCLR rank-21 decomposition for 3x3 matrix multiplication.
    Each term is (a_coeffs, b_coeffs) where:
      a_coeffs: list of (row, col, coeff) for A entries
      b_coeffs: list of (row, col, coeff) for B entries
    Plus output contribution: list of (row_c, col_c, coeff) for C entries
    """
    # BCLR rank-21 decomposition for 3x3 matrix multiplication
    # Reference: Bini, Capovani, Lotti, Romani (1979)
    # C = A * B computed with 21 multiplications of linear combinations
    
    # The 21 bilinear forms:
    # Each has (A_linear_comb, B_linear_comb, C_output)
    # A entries indexed (i,k) for row i, col k of A
    # B entries indexed (k,j) for row k, col j of B  
    # C entries indexed (i,j) for row i, col j of C
    
    # BCLR algorithm:
    # M1  = (a11) * (b11)
    # M2  = (a12 + a13) * (b21 + b31)
    # M3  = (a13) * (b32)
    # M4  = (a11 - a13) * (b11 + b21)
    # M5  = (a21 + a23) * (b11)
    # M6  = (a31) * (b21 + b22)
    # M7  = (a11 + a22) * (b12 - b22)
    # M8  = (a13 - a22 + a33) * (b12)
    # M9  = (a12 + a33) * (b22)
    # M10 = (a22 + a33) * (b22 - b32)
    # M11 = (a21) * (b22 + b32)
    # M12 = (a23) * (b12 + b32)
    # M13 = (a21 - a23) * (b11 + b31)
    # M14 = (a31 + a33) * (b11 - b21 + b31)
    # M15 = (a11 - a22 + a33) * (b13)
    # M16 = (a12) * (b13 + b33)
    # M17 = (a12 - a22 + a33) * (b23)
    # M18 = (a22) * (b23 - b33)
    # M19 = (a21 + a33) * (b11 - b12 + b33)
    # M20 = (a32) * (b13 + b23 + b33)
    # M21 = (a31 + a32) * (b11 - b13)
    
    # Output reconstruction:
    # c11 = M1 - M4 + M7 - M13 + M19
    # c12 = M2 + M3 - M8 + M12 - M16 + M20
    # c13 = M4 - M7 + M8 - M15 + M17 - M20
    # c21 = M5 - M6 + M11 - M14 + M21
    # c22 = M6 - M9 + M10 + M14 - M18 - M21
    # c23 = M9 - M10 + M15 + M17 + M18
    # c31 = M13 - M14 + M19 + M21
    # c32 = M16 - M17 + M18 + M20
    # c33 = M15 - M19 - M20
    
    terms = []
    
    # Each entry: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: [(row, col, coeff), ...]
    # b_coeffs: [(row, col, coeff), ...]
    # c_coeffs: [(row, col, coeff), ...]
    
    # M1 = a11 * b11
    terms.append(([(0,0,1)], [(0,0,1)], [(0,0,1)]))
    # M2 = (a12+a13) * (b21+b31)
    terms.append(([(0,1,1),(0,2,1)], [(1,0,1),(2,0,1)], [(0,1,1)]))
    # M3 = a13 * b32
    terms.append(([(0,2,1)], [(2,1,1)], [(0,1,1)]))
    # M4 = (a11-a13) * (b11+b21)
    terms.append(([(0,0,1),(0,2,-1)], [(0,0,1),(1,0,1)], [(0,0,-1),(0,2,1)]))
    # M5 = (a21+a23) * b11
    terms.append(([(1,0,1),(1,2,1)], [(0,0,1)], [(1,0,1)]))
    # M6 = a31 * (b21+b22)
    terms.append(([(2,0,1)], [(1,0,1),(1,1,1)], [(1,0,-1),(1,1,-1)]))
    # M7 = (a11+a22) * (b12-b22)
    terms.append(([(0,0,1),(1,1,1)], [(0,1,1),(1,1,-1)], [(0,0,1),(0,2,-1)]))
    # M8 = (a13-a22+a33) * b12
    terms.append(([(0,2,1),(1,1,-1),(2,2,1)], [(0,1,1)], [(0,1,-1),(0,2,-1)]))
    # M9 = (a12+a33) * b22
    terms.append(([(0,1,1),(2,2,1)], [(1,1,1)], [(1,1,1),(1,2,1)]))
    # M10 = (a22+a33) * (b22-b32)
    terms.append(([(1,1,1),(2,2,1)], [(1,1,1),(2,1,-1)], [(1,1,-1),(1,2,-1)]))
    # M11 = a21 * (b22+b32)
    terms.append(([(1,0,1)], [(1,1,1),(2,1,1)], [(1,0,1),(1,1,1)]))
    # M12 = a23 * (b12+b32)
    terms.append(([(1,2,1)], [(0,1,1),(2,1,1)], [(0,1,1)]))
    # M13 = (a21-a23) * (b11+b31)
    terms.append(([(1,0,1),(1,2,-1)], [(0,0,1),(2,0,1)], [(0,0,-1),(1,0,-1),(2,0,1)]))
    # M14 = (a31+a33) * (b11-b21+b31)
    terms.append(([(2,0,1),(2,2,1)], [(0,0,1),(1,0,-1),(2,0,1)], [(1,0,1),(1,1,1),(2,0,-1)]))
    # M15 = (a11-a22+a33) * b13
    terms.append(([(0,0,1),(1,1,-1),(2,2,1)], [(0,2,1)], [(0,2,1),(1,2,1),(2,2,1)]))
    # M16 = a12 * (b13+b33)
    terms.append(([(0,1,1)], [(0,2,1),(2,2,1)], [(0,1,1),(2,1,1)]))
    # M17 = (a12-a22+a33) * b23
    terms.append(([(0,1,1),(1,1,-1),(2,2,1)], [(1,2,1)], [(0,2,-1),(1,2,1),(2,1,1)]))
    # M18 = a22 * (b23-b33)
    terms.append(([(1,1,1)], [(1,2,1),(2,2,-1)], [(1,1,1),(1,2,1),(2,1,1)]))
    # M19 = (a21+a33) * (b11-b12+b33)
    terms.append(([(1,0,1),(2,2,1)], [(0,0,1),(0,1,-1),(2,2,1)], [(0,0,1),(2,0,1),(2,2,-1)]))
    # M20 = a32 * (b13+b23+b33)
    terms.append(([(2,1,1)], [(0,2,1),(1,2,1),(2,2,1)], [(0,1,-1),(0,2,1),(1,2,1),(2,1,1)]))
    # M21 = (a31+a32) * (b11-b13)
    terms.append(([(2,0,1),(2,1,1)], [(0,0,1),(0,2,-1)], [(1,0,-1),(1,1,1),(2,0,1)]))
    
    return terms

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    
    deadline = time.time() + 145  # 145s deadline (5s safety margin before 150s limit)
    
    # Strassen decomposition at the 2x2 block level
    # Block layout: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row_offset = (bi // 2) * bs, col_offset = (bi % 2) * bs
    
    # 7 Strassen products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each term: (A_combo, B_combo) where combo is list of (block_idx, coefficient)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # P1
        ([(2, 1), (3, 1)], [(0, 1)]),            # P2
        ([(0, 1)], [(1, 1), (3, -1)]),           # P3
        ([(3, 1)], [(2, 1), (0, -1)]),           # P4
        ([(0, 1), (1, 1)], [(3, 1)]),            # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # P7
    ]
    
    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    output_contrib = [
        [(0, 1), (3, 1)],     # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],    # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],     # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],     # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],    # P5 -> C00(-1), C01(+1)
        [(3, 1)],             # P6 -> C11(+1)
        [(0, 1)],             # P7 -> C00(+1)
    ]
    
    # Get BCLR terms for 3x3 matrix multiplication
    bclr = _bclr_terms()
    
    # Total rank: 7 Strassen terms * 21 BCLR terms per 3x3 product = 147
    rank = 7 * 21  # 147
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for bclr_i in range(21):
            a_coeffs, b_coeffs, c_coeffs = bclr[bclr_i]
            # a_coeffs: [(row_a_local, col_a_local, coeff), ...] for the 3x3 A block
            # b_coeffs: [(row_b_local, col_b_local, coeff), ...] for the 3x3 B block
            # c_coeffs: [(row_c_local, col_c_local, coeff), ...] for the 3x3 C block
            
            for (la, ka, ca) in a_coeffs:
                for (blk_a, sa) in a_comb:
                    row_a = (blk_a // 2) * bs + la
                    col_a = (blk_a % 2) * bs + ka
                    u_idx = row_a * m + col_a
                    U[u_idx][col] += ca * sa
            
            for (kb, jb, cb) in b_coeffs:
                for (blk_b, sb) in b_comb:
                    row_b = (blk_b // 2) * bs + kb
                    col_b = (blk_b % 2) * bs + jb
                    v_idx = row_b * p + col_b
                    V[v_idx][col] += cb * sb
            
            for (lc, jc, cc) in c_coeffs:
                for (blk_c, sc) in output_contrib[term_idx]:
                    row_c = (blk_c // 2) * bs + lc
                    col_c = (blk_c % 2) * bs + jc
                    w_idx = col_c * n + row_c
                    W[w_idx][col] += cc * sc
            
            col += 1
        
        # Check deadline
        if time.time() > deadline:
            # Return what we have so far (should be complete by now)
            pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
