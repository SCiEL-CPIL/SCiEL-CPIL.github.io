"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 140
    
    # We use a Kronecker product approach:
    # 4x4 * 4x7 = (2x2 * 2x4) kron (2x2 * 2x2) roughly
    # More precisely: treat 4x4 * 4x7 as block matrix multiplication
    # where we split into 2x2 blocks.
    #
    # A (4x4) = [[A11, A12], [A21, A22]] each 2x2
    # B (4x7) = [[B11, B12, B13], [B21, B22, B23]] where B11,B21 are 2x2, B12,B22 are 2x2, B13,B23 are 2x3
    #
    # Actually, let me think of it differently.
    # Split the 4x4 * 4x7 problem using the fact that 4 = 2*2 and 7 = 2*2 + 3
    #
    # Alternative: Use the known rank-7 decomposition for 2x2*2x2 (Strassen)
    # and combine it with schoolbook for the remaining parts.
    #
    # For 4x4 * 4x4: rank 28 via Strassen (7 mults of 2x2, each 2x2 has rank 4... no)
    # Actually Strassen gives 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 4 (schoolbook) or can be lower.
    # But the tensor rank of 2x2*2x2 is 7 (Strassen).
    # So 4x4*4x4 via Strassen has rank 7*4 = 28? No, that's not right either.
    #
    # Let me think more carefully. The tensor for 2x2*2x2 multiplication has rank 7.
    # The tensor for 4x4*4x4 can be decomposed using Kronecker products.
    #
    # Actually, the simplest correct approach:
    # 4x4 * 4x7: We can view this as 4x4 * (4x4 + 4x3)
    # = (4x4 * 4x4) + (4x4 * 4x3)
    #
    # For 4x4 * 4x4: Use Strassen-like decomposition
    # Split A and B into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # Using Strassen's 7 multiplications:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 * 2x2 product, which has tensor rank 7 (using Strassen recursively)
    # or rank 4 (using schoolbook for 2x2).
    #
    # If we use schoolbook for the 2x2 multiplications:
    # Each Mi contributes 4 terms, total 7*4 = 28 terms for the 4x4*4x4 part.
    #
    # For 4x4 * 4x3:
    # Split B into B_left (4x2) and B_right (4x1)
    # Actually 4x3: split into (4x2) + (4x1)
    # 4x4 * 4x2: similar to above but with 2 columns
    # 4x4 * 4x1: schoolbook gives 16 terms
    #
    # Let me just build the full decomposition directly.
    
    # I'll use a systematic approach:
    # The multiplication tensor T[i,j,k] for A(4x4)*B(4x7)->C(4x7)
    # has T[i,j,k] = 1 for all valid (i,j,k), meaning C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Rank-1 terms: each term selects specific rows of A, columns of B, and outputs of C
    #
    # Let me build using the block Strassen approach for the 4x4*4x4 part
    # and schoolbook for the 4x4*4x3 part.
    
    terms = []  # Each: (i, j, k, u_coeff, v_coeff, w_coeff)
    
    def add_term(i, j, k, uc, vc, wc):
        terms.append((i, j, k, uc, vc, wc))
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    # Using Strassen with 2x2 blocks
    
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # M1 = (A11+A22)*(B11+B22)
    # (A11+A22)[di,dj] = A[di,dj] + A[di+2,dj+2]
    # (B11+B22)[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    # M1[di,dk] = sum_dj (A[di,dj]+A[di+2,dj+2])*(B[dj,dk]+B[dj+2,dk+2])
    # = sum_dj A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2]
    
    # M1 contributes +1 to C11 (=C[0:2,0:2]) and +1 to C22 (=C[2:4,2:4])
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Term: A[di,dj]*B[dj,dk] -> C[di,dk] (C11)
                add_term(di, dj, dk, 1, 1, 1)
                # Term: A[di,dj]*B[dj+2,dk+2] -> C[di,dk] (C11)
                add_term(di, dj, dk+2, 1, 1, 1)
                # Term: A[di+2,dj+2]*B[dj,dk] -> C[di,dk] (C11)
                add_term(di+2, dj+2, dk, 1, 1, 1)
                # Term: A[di+2,dj+2]*B[dj+2,dk+2] -> C[di,dk] (C11)
                add_term(di+2, dj+2, dk+2, 1, 1, 1)
                # Same terms but contributing to C22 = C[di+2,dk+2]
                add_term(di, dj, dk, 1, 1, 1)
                add_term(di, dj, dk+2, 1, 1, 1)
                add_term(di+2, dj+2, dk, 1, 1, 1)
                add_term(di+2, dj+2, dk+2, 1, 1, 1)
    
    # Wait, this is wrong. The issue is that each "term" in the tensor decomposition
    # is a single rank-1 contribution. The Strassen algorithm works by combining
    # multiple (i,j,k) triples with the same coefficient pattern.
    
    # Let me restart with a cleaner approach.
    
    terms = []
    
    # For the 4x4 * 4x4 part, I'll directly enumerate the Strassen terms.
    # Each Strassen multiplication M = X*Y where X is a linear combination of A-blocks
    # and Y is a linear combination of B-blocks produces a 2x2 result.
    # The 2x2 result M[di,dk] = sum_dj X[di,dj]*Y[dj,dk]
    # Each such term corresponds to a specific (i,j,k) in the original tensor.
    
    # Let me define helper functions to add Strassen terms.
    
    def add_strassen_term(a_indices, b_indices, c_row_offset, c_col_offset, coeff):
        """
        a_indices: list of (row, col, sign) for the A-block combination
        b_indices: list of (row, col, sign) for the B-block combination  
        c_row_offset, c_col_offset: offset of the C-block this contributes to
        coeff: overall coefficient for this Strassen multiplication
        """
        for (ar, ac, asgn) in a_indices:
            for (br, bc, bsgn) in b_indices:
                # A[ar+di, ac+dj] * B[br+dj, bc+dk] for di,dj,dk in 0,1
                # contributes to C[c_row_offset+di, c_col_offset+dk]
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            i = ar + di
                            j = ac + dj  # = br + dj (must match)
                            k = bc + dk
                            uc = asgn
                            vc = bsgn
                            wc = coeff
                            add_term(i, j, k, uc, vc, wc)
    
    # M1 = (A11+A22)*(B11+B22)
    # A11: rows 0-1, cols 0-1; A22: rows 2-3, cols 2-3
    # B11: rows 0-1, cols 0-1; B22: rows 2-3, cols 2-3
    # Contributes +1 to C11 (offset 0,0) and +1 to C22 (offset 2,2)
    a1 = [(0, 0, 1), (2, 2, 1)]
    b1 = [(0, 0, 1), (2, 2, 1)]
    add_strassen_term(a1, b1, 0, 0, 1)   # to C11
    add_strassen_term(a1, b1, 2, 2, 1)   # to C22
    
    # M2 = (A21+A22)*B11
    # A21: rows 2-3, cols 0-1; A22: rows 2-3, cols 2-3
    # B11: rows 0-1, cols 0-1
    # Contributes +1 to C21 (offset 2,0)
    a2 = [(2, 0, 1), (2, 2, 1)]
    b2 = [(0, 0, 1)]
    add_strassen_term(a2, b2, 2, 0, 1)   # to C21
    
    # M3 = A11*(B12-B22)
    # A11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3; B22: rows 2-3, cols 2-3
    # Contributes +1 to C12 (offset 0,2)
    a3 = [(0, 0, 1)]
    b3 = [(0, 2, 1), (2, 2, -1)]
    add_strassen_term(a3, b3, 0, 2, 1)   # to C12
    
    # M4 = A22*(B21-B11)
    # A22: rows 2-3, cols 2-3
    # B21: rows 2-3, cols 0-1; B11: rows 0-1, cols 0-1
    # Contributes +1 to C11 (offset 0,0) and +1 to C21 (offset 2,0)
    a4 = [(2, 2, 1)]
    b4 = [(2, 0, 1), (0, 0, -1)]
    add_strassen_term(a4, b4, 0, 0, 1)   # to C11
    add_strassen_term(a4, b4, 2, 0, 1)   # to C21
    
    # M5 = (A11+A12)*B22
    # A11: rows 0-1, cols 0-1; A12: rows 0-1, cols 2-3
    # B22: rows 2-3, cols 2-3
    # Contributes -1 to C11 (offset 0,0) and +1 to C12 (offset 0,2)
    a5 = [(0, 0, 1), (0, 2, 1)]
    b5 = [(2, 2, 1)]
    add_strassen_term(a5, b5, 0, 0, -1)  # to C11
    add_strassen_term(a5, b5, 0, 2, 1)   # to C12
    
    # M6 = (A21-A11)*(B11+B12)
    # A21: rows 2-3, cols 0-1; A11: rows 0-1, cols 0-1
    # B11: rows 0-1, cols 0-1; B12: rows 0-1, cols 2-3
    # Contributes +1 to C22 (offset 2,2)
    a6 = [(2, 0, 1), (0, 0, -1)]
    b6 = [(0, 0, 1), (0, 2, 1)]
    add_strassen_term(a6, b6, 2, 2, 1)   # to C22
    
    # M7 = (A12-A22)*(B21+B22)
    # A12: rows 0-1, cols 2-3; A22: rows 2-3, cols 2-3
    # B21: rows 2-3, cols 0-1; B22: rows 2-3, cols 2-3
    # Contributes +1 to C11 (offset 0,0)
    a7 = [(0, 2, 1), (2, 2, -1)]
    b7 = [(2, 0, 1), (2, 2, 1)]
    add_strassen_term(a7, b7, 0, 0, 1)   # to C11
    
    # Part 2: 4x4 * 4x3 (B columns 4-6) using schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                add_term(i, j, k, 1, 1, 1)
    
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (i, j, k, uc, vc, wc) in enumerate(terms):
        U[i * m + j][r] = uc
        V[j * p + k][r] = vc
        W[k * n + i][r] = wc
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
