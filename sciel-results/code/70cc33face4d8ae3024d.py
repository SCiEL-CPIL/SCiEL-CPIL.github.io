import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 77
    rank = 77
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen decomposition for 2x2x2
    # Returns lists of (coeff_u, coeff_v, coeff_w) for the 7 terms
    # u: 2x2, v: 2x2, w: 2x2
    def strassen_2x2x2():
        terms = []
        # M1
        u = [[1, 0], [0, 1]]
        v = [[1, 0], [0, 1]]
        w = [[1, 0], [0, 0]] # C00 += M1
        terms.append((u, v, [[1, 0], [0, 0]])) # w is C coeff
        # Actually, w should be the coefficient for C.
        # C00 gets M1. So w[0,0]=1.
        
        # Let's just store the linear forms
        # Term k: U_part, V_part, W_part
        # U_part is 2x2 matrix of coeffs for A
        # V_part is 2x2 matrix of coeffs for B
        # W_part is 2x2 matrix of coeffs for C
        
        T = []
        # 1
        T.append(([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,0]]))
        # 2
        T.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]])) # C10 += M2
        # 3
        T.append(([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]])) # C01 += M3
        # 4
        T.append(([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[1,0]])) # C00 += M4, C10 += M4
        # 5
        T.append(([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]])) # C00 -= M5, C01 += M5
        # 6
        T.append(([[ -1,0],[1,0]], [[1,1],[0,0]], [[0,1],[0,0]])) # C11 += M6
        # 7
        T.append(([[0,1],[0,-1]], [[0,1],[1,1]], [[1,0],[0,0]])) # C00 += M7
        
        return T

    strassen_terms = strassen_2x2x2()
    
    # We have 11 sub-multiplications of 2x2x2
    # 7 from Strassen on first 2 block columns of B
    # 4 from Schoolbook on 3rd block column of B
    
    # Global indices:
    # A is 4x4. Blocks A11(0,0), A12(0,2), A21(2,0), A22(2,2)
    # B is 4x6. Blocks B11(0,0), B12(0,2), B13(0,4), B21(2,0), B22(2,2), B23(2,4)
    # C is 6x4 (p x n). Blocks C11(0,0), C12(2,0), C13(4,0), C21(0,2), C22(2,2), C23(4,2)
    # Note: W index is k*n + i. C is 6x4.
    # C row k (0..5), col i (0..3).
    # Block C11 is rows 0,1 cols 0,1.
    # Block C12 is rows 2,3 cols 0,1.
    # Block C13 is rows 4,5 cols 0,1.
    # Block C21 is rows 0,1 cols 2,3.
    # Block C22 is rows 2,3 cols 2,3.
    # Block C23 is rows 4,5 cols 2,3.
    
    col = 0
    
    def add_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col
        for u, v, w in strassen_terms:
            # u is 2x2 coeffs for A block
            for i in range(2):
                for j in range(2):
                    if u[i][j] != 0:
                        U[a_row_start + i, a_col_start + j, col] = u[i][j]
                        
            # v is 2x2 coeffs for B block
            for j in range(2):
                for k in range(2):
                    if v[j][k] != 0:
                        V[b_row_start + j, b_col_start + k, col] = v[j][k]
                        
            # w is 2x2 coeffs for C block
            for k in range(2):
                for i in range(2):
                    if w[k][i] != 0:
                        W[c_row_start + k, c_col_start + i, col] = w[k][i]
                        
            col += 1

    # Strassen Part:
    # M1 = (A11+A22)(B11+B21) -> C11+=, C21+=
    # A blocks: A11(0,0), A22(2,2)
    # B blocks: B11(0,0), B21(2,0)
    # C blocks: C11(0,0), C21(0,2)
    
    # We need to combine A11 and A22 into one "virtual" A block?
    # No, the decomposition adds to U.
    # The term is (A11+A22) * (B11+B21).
    # This expands to A11B11 + A11B21 + A22B11 + A22B21.
    # But Strassen uses the product of sums.
    # The rank-1 term is u * v * w.
    # u = A11 + A22.
    # So in U, we put the coeffs for A11 and A22.
    
    # Let's refine the add function to accept multiple A blocks and B blocks.
    
    def add_strassen_term(a_blocks, b_blocks, c_blocks, coeff_sign=1):
        # a_blocks: list of (row_start, col_start, sign)
        # b_blocks: list of (row_start, col_start, sign)
        # c_blocks: list of (row_start, col_start, sign)
        pass # This is getting complicated.

    # Simpler: Just iterate through the 7 Strassen formulas for the block level.
    # For each formula, we identify the A blocks and B blocks involved.
    # Then we call the 2x2x2 decomposition for each pair?
    # No, the 2x2x2 decomposition is for the *entries* of the blocks.
    # The Strassen formula is at the block level.
    # M1 = (A11+A22)(B11+B21).
    # This is a product of a 2x2 matrix (sum of blocks) and a 2x2 matrix (sum of blocks).
    # This product is a 2x2x2 multiplication.
    # We can decompose this product using the 7 Strassen terms for 2x2x2.
    # The inputs to this 2x2x2 multiplication are the entries of (A11+A22) and (B11+B21).
    # So U will contain coeffs for A11 and A22.
    
    # Let's define the 7 block-level Strassen terms for the 4x4x4 part (first 2 cols of B).
    # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2)
    # B blocks (cols 0,1): B11(0,0), B12(0,2), B21(2,0), B22(2,2)
    # C blocks (cols 0,1): C11(0,0), C12(0,2), C21(2,0), C22(2,2)
    
    # Term 1: (A11+A22)(B11+B21)
    # A: A11(+1), A22(+1)
    # B: B11(+1), B21(+1)
    # C: C11(+1), C21(+1)
    
    # We need to add the 2x2x2 decomposition for this term.
    # The "A" input to the 2x2x2 decomposer is the sum of blocks.
    # So in U, for each entry of the 2x2 block, we add the coeff.
    
    def add_block_term(a_specs, b_specs, c_specs):
        nonlocal col
        for u, v, w in strassen_terms:
            for i in range(2):
                for j in range(2):
                    if u[i][j] != 0:
                        for a_rs, a_cs, sign in a_specs:
                            U[a_rs + i, a_cs + j, col] += u[i][j] * sign
            for j in range(2):
                for k in range(2):
                    if v[j][k] != 0:
                        for b_rs, b_cs, sign in b_specs:
                            V[b_rs + j, b_cs + k, col] += v[j][k] * sign
            for k in range(2):
                for i in range(2):
                    if w[k][i] != 0:
                        for c_rs, c_cs, sign in c_specs:
                            W[c_rs + k, c_cs + i, col] += w[k][i] * sign
            col += 1

    # Block Strassen Terms for C[:, 0:4]
    # 1
    add_block_term([(0,0,1), (2,2,1)], [(0,0,1), (2,0,1)], [(0,0,1), (0,2,1)])
    # 2
    add_block_term([(2,0,1), (2,2,1)], [(0,0,1), (0,2,1)], [(2,0,1), (2,2,1)])
    # 3
    add_block_term([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1)])
    # 4
    add_block_term([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)])
    # 5
    add_block_term([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (2,2,1)])
    # 6
    add_block_term([(2,0,-1), (0,0,1)], [(0,0,1), (0,2,1)], [(2,2,1)])
    # 7
    add_block_term([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1)])
    
    # Schoolbook Part for C[:, 4:6]
    # C13 = A11 B13 + A12 B23
    # C23 = A21 B13 + A22 B23
    # B13 is (0,4), B23 is (2,4)
    # C13 is (4,0), C23 is (4,2)
    
    # Term 1: A11 B13 -> C13
    add_block_term([(0,0,1)], [(0,4,1)], [(4,0,1)])
    # Term 2: A12 B23 -> C13
    add_block_term([(0,2,1)], [(2,4,1)], [(4,0,1)])
    # Term 3: A21 B13 -> C23
    add_block_term([(2,0,1)], [(0,4,1)], [(4,2,1)])
    # Term 4: A22 B23 -> C23
    add_block_term([(2,2,1)], [(2,4,1)], [(4,2,1)])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
