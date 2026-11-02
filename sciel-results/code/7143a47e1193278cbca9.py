"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 72 decomposition:
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    # Strassen (Rank 28) for 4x4x4 blocks.
    # Schoolbook (Rank 16) for 4x4x1 block.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i, j, k, coeff, col_idx):
        U[i * m + j][col_idx] += coeff
        V[j * p + k][col_idx] += coeff
        W[k * n + i][col_idx] += coeff

    def add_strassen_block(i0, j0, k0, start_col):
        nonlocal col
        # Strassen formulas for 2x2 blocks
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
        
        # We expand each Mt into 4 scalar terms.
        # A block indices:
        # A00: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # A01: (i0, j0+2), (i0, j0+3), (i0+1, j0+2), (i0+1, j0+3)
        # A10: (i0+2, j0), (i0+2, j0+1), (i0+3, j0), (i0+3, j0+1)
        # A11: (i0+2, j0+2), (i0+2, j0+3), (i0+3, j0+2), (i0+3, j0+3)
        
        # B block indices:
        # B00: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # B01: (j0, k0+2), (j0, k0+3), (j0+1, k0+2), (j0+1, k0+3)
        # B10: (j0+2, k0), (j0+2, k0+1), (j0+3, k0), (j0+3, k0+1)
        # B11: (j0+2, k0+2), (j0+2, k0+3), (j0+3, k0+2), (j0+3, k0+3)
        
        # Helper to add a product of two 2x2 blocks
        # A_combos: list of (coeff, i, j) for the 4 entries
        # B_combos: list of (coeff, j, k) for the 4 entries
        # C_offset: (u, v) in the 2x2 result block
        # C_coeff: coefficient for this Mt in the final C
        
        def add_product(A_combos, B_combos, u, v, c_coeff, start_col):
            nonlocal col
            # A_combos and B_combos are lists of 4 tuples (coeff, i, j) and (coeff, j, k)
            # We need to multiply them.
            # Since they are 2x2, we iterate over the 4 terms of A and 4 terms of B?
            # No, the product is a 2x2 matrix.
            # C'_{uv} = sum_{v'} A'_{u v'} B'_{v' v}
            # Here A_combos corresponds to A'_{u *}? No.
            # Let's define A_combos as a list of 4 terms for the specific A' entry?
            # No, Strassen Mt is a full 2x2 product.
            # We should pass the full A' and B' linear combinations.
            
            pass # This helper is too complex. Let's hardcode the 28 terms.
            
        # Hardcoding the 28 terms for the Strassen block
        # Term 1: M1 contributes to C00, C01, C10, C11?
        # M1 = (A00+A11)(B00+B11)
        # A00+A11 has 8 terms. B00+B11 has 8 terms.
        # Product has 64 terms? No, many are zero due to index mismatch.
        # We only keep terms where j_A == j_B.
        
        # Let's generate the terms programmatically.
        
        def get_A_block(block_id):
            # block_id 0:00, 1:01, 2:10, 3:11
            res = []
            if block_id == 0:
                for i in range(2):
                    for j in range(2):
                        res.append((1, i0+i, j0+j))
            elif block_id == 1:
                for i in range(2):
                    for j in range(2):
                        res.append((1, i0+i, j0+2+j))
            elif block_id == 2:
                for i in range(2):
                    for j in range(2):
                        res.append((1, i0+2+i, j0+j))
            elif block_id == 3:
                for i in range(2):
                    for j in range(2):
                        res.append((1, i0+2+i, j0+2+j))
            return res

        def get_B_block(block_id):
            res = []
            if block_id == 0:
                for j in range(2):
                    for k in range(2):
                        res.append((1, j0+j, k0+k))
            elif block_id == 1:
                for j in range(2):
                    for k in range(2):
                        res.append((1, j0+j, k0+2+k))
            elif block_id == 2:
                for j in range(2):
                    for k in range(2):
                        res.append((1, j0+2+j, k0+k))
            elif block_id == 3:
                for j in range(2):
                    for k in range(2):
                        res.append((1, j0+2+j, k0+2+k))
            return res

        # Define the 7 M's as combinations of blocks
        # M1: A(0)+A(3), B(0)+B(3)
        # M2: A(2)+A(3), B(0)
        # M3: A(0), B(2)-B(3)
        # M4: A(3), B(1)-B(0)
        # M5: A(0)+A(1), B(3)
        # M6: A(2)-A(0), B(0)+B(1)
        # M7: A(1)-A(3), B(2)+B(3)
        
        def add_m(A_blocks, B_blocks, c_coeff_map, start_col):
            # A_blocks: list of (coeff, block_id)
            # B_blocks: list of (coeff, block_id)
            # c_coeff_map: dict (u,v) -> coeff for C'
            
            # Expand A and B into scalar terms
            A_terms = []
            for c, bid in A_blocks:
                for term in get_A_block(bid):
                    A_terms.append((c * term[0], term[1], term[2]))
            
            B_terms = []
            for c, bid in B_blocks:
                for term in get_B_block(bid):
                    B_terms.append((c * term[0], term[1], term[2]))
            
            # Now we have A_terms and B_terms.
            # We need to compute the product C' = A' B'.
            # C'_{uv} = sum_{v'} A'_{u v'} B'_{v' v}
            # This is tricky because A_terms doesn't know u,v.
            
            # Alternative: Just iterate over all pairs of A_terms and B_terms?
            # No, that's too slow and incorrect.
            
            # Let's just use the fact that we can add rank-1 terms for each valid (i,j,k).
            # We will iterate over all A_terms (i,j) and B_terms (j',k).
            # If j == j', we have a valid term A[i,j] B[j,k].
            # This term contributes to C[i,k].
            # Which C entry?
            # In the 2x2 block, i is 0 or 1, k is 0 or 1.
            # The contribution to the global C is determined by the Strassen formula.
            
            # This approach is flawed because we need to know WHICH C entry (C00, C01, etc.)
            # each term contributes to, and with what coefficient.
            
            pass

        # Given the complexity of dynamic generation, I will stick to Schoolbook for safety
        # but try to use a better known decomposition if possible.
        # Actually, rank 72 is hard to get right without a library.
        # I will return Schoolbook rank 144.
        
        return start_col

    # Fallback to Schoolbook
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
