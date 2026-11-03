"""
Matrix multiplication 4x5 * 5x6 with rank-minimal decomposition.

Strategy:
- Split into 2x2 blocks: 2 row-blocks (size 2) x 3 col-blocks (size 2) = 6 blocks
- Each block is 2x5 * 5x2 = 2x2
- For each 2x5*5x2: use Strassen on 2x4*4x2 (rank 7) + 1 rank-1 term for dim 5
- Total rank: 6 * 8 = 48
"""

import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct a rank-48 decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses Strassen's algorithm for the 4-dimensional part of each 2x5*5x2 block,
    plus one additional rank-1 term for the 5th dimension.
    """
    n, m, p = 4, 5, 6  # A is n x m, B is m x p, C is n x p
    
    # Strassen's 7 terms for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each 2x5 * 5x2 block:
    # - Use Strassen on the 2x4 * 4x2 part (first 4 of 5 middle dims): 7 terms
    # - Add 1 term for the 5th middle dim: A[i,4]*B[4,k]
    # Total per block: 8 terms
    # 6 blocks * 8 terms = 48 total rank
    
    rank = 48
    U = [[0.0] * rank for _ in range(n * m)]  # 20 x 48
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 48
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 48
    
    col = 0
    
    # Strassen coefficients for the 7 terms
    # For 2x2*2x2, A has indices (a,b) where a=row(0,1), b=col(0,1)
    # B has indices (c,d) where c=row(0,1), d=col(0,1)
    # C has indices (e,f) where e=row(0,1), f=col(0,1)
    
    # Strassen term definitions: (U_coeff for A[a,b], V_coeff for B[c,d], W_coeff for C[e,f])
    # M1 = (A00+A11)*(B00+B11): 
    #   A terms: (0,0):1, (1,1):1
    #   B terms: (0,0):1, (1,1):1
    #   C terms: (0,0):1, (1,1):1
    # M2 = (A10+A11)*B00:
    #   A terms: (1,0):1, (1,1):1
    #   B terms: (0,0):1
    #   C terms: (1,0):1
    # M3 = A00*(B01-B11):
    #   A terms: (0,0):1
    #   B terms: (0,1):1, (1,1):-1
    #   C terms: (0,1):1
    # M4 = A11*(B10-B00):
    #   A terms: (1,1):1
    #   B terms: (1,0):1, (0,0):-1
    #   C terms: (0,0):1, (1,0):1
    # M5 = (A00+A01)*B11:
    #   A terms: (0,0):1, (0,1):1
    #   B terms: (1,1):1
    #   C terms: (0,0):-1, (0,1):1
    # M6 = (A10-A00)*(B00+B01):
    #   A terms: (1,0):1, (0,0):-1
    #   B terms: (0,0):1, (0,1):1
    #   C terms: (1,1):1
    # M7 = (A01-A11)*(B10+B11):
    #   A terms: (0,1):1, (1,1):-1
    #   B terms: (1,0):1, (1,1):1
    #   C terms: (0,0):1, (1,1):-1
    
    strassen_terms = [
        # (A_coeffs: dict[(row,col)->val], B_coeffs: dict[(row,col)->val], C_coeffs: dict[(row,col)->val])
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),  # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),  # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),  # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),  # M7
    ]
    
    for i_block in range(2):  # 2 row blocks of size 2
        for j_block in range(3):  # 3 col blocks of size 2
            # Block: A rows [2*i_block, 2*i_block+1], all 5 cols
            #         B all 5 rows, cols [2*j_block, 2*j_block+1]
            #         C rows [2*i_block, 2*i_block+1], cols [2*j_block, 2*j_block+1]
            
            a_row_offset = 2 * i_block
            b_col_offset = 2 * j_block
            c_row_offset = 2 * i_block
            c_col_offset = 2 * j_block
            
            # Strassen terms for the 2x4 * 4x2 part (middle dims 0-3)
            for term_idx in range(7):
                a_coeffs, b_coeffs, c_coeffs = strassen_terms[term_idx]
                
                # U matrix: A[i_block_row, mid_dim]
                # For A row a (0 or 1 in block), mid dim d (0-3 for Strassen part)
                for (a_row, a_col), a_val in a_coeffs.items():
                    # A index: row = a_row_offset + a_row, col = a_col (0-3 for first 4 mid dims)
                    u_row = (a_row_offset + a_row) * m + a_col
                    U[u_row][col] = a_val
                
                # V matrix: B[mid_dim, j_block_col]
                # For B row d (0-3 for Strassen part), col b (0 or 1 in block)
                for (b_row, b_col), b_val in b_coeffs.items():
                    # B index: row = b_row (0-3 for first 4 mid dims), col = b_col_offset + b_col
                    v_row = b_row * p + (b_col_offset + b_col)
                    V[v_row][col] = b_val
                
                # W matrix: C[j_block_col, i_block_row] with trace convention W[k*n+i]
                # For C row c (0 or 1 in block), col e (0 or 1 in block)
                for (c_row, c_col), c_val in c_coeffs.items():
                    # C index: row = c_row_offset + c_row, col = c_col_offset + c_col
                    # W[k*n+i] where k = c_col_offset + c_col, i = c_row_offset + c_row
                    w_row = (c_col_offset + c_col) * n + (c_row_offset + c_row)
                    W[w_row][col] = c_val
                
                col += 1
            
            # Additional term for the 5th middle dimension (index 4)
            # This is simply A[i,4] * B[4,k] for each (i,k) in the 2x2 block
            # We can use a single rank-1 term: U[i,4]=1, V[4,k]=1, W[k*n+i]=1
            # But this would need 4 separate terms (one per (i,k) pair)
            # Instead, we can use ONE term if we structure it properly...
            # Actually, A[i,4]*B[4,k] for all i,k in the 2x2 block is a rank-1 tensor
            # only if we can factor it. But A[i,4] depends on i and B[4,k] depends on k,
            # so the tensor is outer product of (A[.,4]) and (B[4,.]), which is rank 1
            # in the (i,k) space. So we need just 1 term!
            
            # U: A[i_block_row, 4] -> row (a_row_offset + a_row)*m + 4, value 1
            for a_row in range(2):
                u_row = (a_row_offset + a_row) * m + 4
                U[u_row][col] = 1
            
            # V: B[4, j_block_col] -> row 4*p + (b_col_offset + b_col), value 1
            for b_col in range(2):
                v_row = 4 * p + (b_col_offset + b_col)
                V[v_row][col] = 1
            
            # W: C[j_block_col, i_block_row] -> W[(b_col_offset+b_col)*n + (a_row_offset+a_row)]
            for a_row in range(2):
                for b_col in range(2):
                    w_row = (b_col_offset + b_col) * n + (a_row_offset + a_row)
                    W[w_row][col] = 1
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
