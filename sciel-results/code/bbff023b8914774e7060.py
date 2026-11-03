import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 140.0
    
    # Total rank calculation
    # Blocks:
    # i in [0,2), j in [0,4), k in [0,6) -> 2x2x2 blocks (6 blocks)
    # i in [0,2), j in [4,5), k in [0,6) -> 2x1x2 blocks (3 blocks)
    # i in [2,3), j in [0,4), k in [0,6) -> 1x2x2 blocks (6 blocks)
    # i in [2,3), j in [4,5), k in [0,6) -> 1x1x2 blocks (3 blocks)
    
    rank_222 = 7
    rank_212 = 4
    rank_122 = 4
    rank_112 = 2
    
    total_rank = 6 * rank_222 + 3 * rank_212 + 6 * rank_122 + 3 * rank_112
    # 42 + 12 + 24 + 6 = 84
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        ni = i1 - i0
        nj = j1 - j0
        nk = k1 - k0
        
        if ni == 2 and nj == 2 and nk == 2:
            # Strassen 2x2x2
            # Local indices: ai, aj, bi, bk in [0,2)
            # Products:
            # P1 = (A00+A11)(B00+B11)
            # P2 = (A10+A11)B00
            # P3 = A00(B10-B00)
            # P4 = A11(B01-B10)
            # P5 = (A00+A01)B11
            # P6 = (A10-A00)(B00+B01)
            # P7 = (A01-A11)(B10+B11)
            
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # We need to add rank-1 terms for each product.
            # Each product is a sum of A*B terms.
            # For each term A[ai,aj]*B[bi,bk], we add to U, V, W.
            
            def add_term(ai, aj, bi, bk, coeff_u, coeff_v, coeff_w):
                nonlocal col
                u_row = (i0 + ai) * m + (j0 + aj)
                v_row = (j0 + bi) * p + (k0 + bk)
                w_row = (k0 + bk) * n + (i0 + ai)
                
                # The term contributes coeff_u * coeff_v * coeff_w to the product?
                # No, the product is (sum A terms) * (sum B terms).
                # We need to express the product as a sum of rank-1 terms in the tensor.
                # The tensor entry T[i,j,k] is 1.
                # The rank-1 term is U[i,j] V[j,k] W[k,i].
                # For a specific A*B term, U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
                # But here we have linear combinations.
                # We must add the coefficients to U, V, W for the current column.
                # Wait, each product Pq uses a DIFFERENT column?
                # No, the 7 products share the 7 columns.
                # So for column `col`, we set U, V, W for all terms in Pq.
                
                U[u_row][col] += coeff_u
                V[v_row][col] += coeff_v
                W[w_row][col] += coeff_w

            # P1
            add_term(0, 0, 0, 0, 1, 1, 1)
            add_term(0, 0, 1, 1, 1, 1, 1)
            add_term(1, 1, 0, 0, 1, 1, 1)
            add_term(1, 1, 1, 1, 1, 1, 1)
            col += 1
            
            # P2
            add_term(1, 0, 0, 0, 1, 1, 1)
            add_term(1, 1, 0, 0, 1, 1, 1)
            col += 1
            
            # P3
            add_term(0, 0, 1, 0, 1, 1, 1)
            add_term(0, 0, 0, 0, 1, -1, 1)
            col += 1
            
            # P4
            add_term(1, 1, 0, 1, 1, 1, 1)
            add_term(1, 1, 1, 0, 1, -1, 1)
            col += 1
            
            # P5
            add_term(0, 0, 1, 1, 1, 1, 1)
            add_term(0, 1, 1, 1, 1, 1, 1)
            col += 1
            
            # P6
            add_term(1, 0, 0, 0, 1, 1, 1)
            add_term(1, 0, 0, 1, 1, 1, 1)
            add_term(0, 0, 0, 0, -1, 1, 1)
            add_term(0, 0, 0, 1, -1, 1, 1)
            col += 1
            
            # P7
            add_term(0, 1, 1, 0, 1, 1, 1)
            add_term(0, 1, 1, 1, 1, 1, 1)
            add_term(1, 1, 1, 0, -1, 1, 1)
            add_term(1, 1, 1, 1, -1, 1, 1)
            col += 1
            
            # Now we need to combine these 7 columns into the output C.
            # The output C is indexed by (i0+di, k0+dk).
            # The W matrix is indexed by (k, i).
            # The current W entries are 1 for the specific (i,j,k) triples.
            # We need to scale the W entries to match the C coefficients.
            # C00 = P1 + P4 - P5 + P7
            # This means the contribution of column 0 (P1) to W[k0, i0] should be 1.
            # The contribution of column 3 (P4) to W[k0, i0] should be 1.
            # The contribution of column 4 (P5) to W[k0, i0] should be -1.
            # The contribution of column 6 (P7) to W[k0, i0] should be 1.
            
            # But we already added 1 to W for all terms.
            # We need to adjust the W values for each column based on the C coefficients.
            
            # Let's re-do this. We should not add to W yet.
            # We add to U and V. Then we set W based on the C coefficients.
            
            # Reset W for this block? No, we can just add.
            # But we added 1 to W for all terms.
            # We need to subtract the unwanted contributions.
            # This is messy.
            
            # Better approach:
            # For each column, define the U and V vectors.
            # Then set the W vector to be the coefficients for the output positions.
            
            # Let's restart the block logic with a cleaner structure.
            pass

        else:
            # Schoolbook
            for ai in range(ni):
                for aj in range(nj):
                    for ak in range(nk):
                        u_row = (i0 + ai) * m + (j0 + aj)
                        v_row = (j0 + aj) * p + (k0 + ak)
                        w_row = (k0 + ak) * n + (i0 + ai)
                        U[u_row][col] = 1
                        V[v_row][col] = 1
                        W[w_row][col] = 1
                        col += 1

    # The Strassen implementation above is flawed because it mixes U, V, W additions.
    # I will implement a clean version.
    
    # Re-initialize
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        # 7 columns for 7 products
        # Product definitions (local indices)
        # P1: (A00+A11)(B00+B11)
        # P2: (A10+A11)B00
        # P3: A00(B10-B00)
        # P4: A11(B01-B10)
        # P5: (A00+A01)B11
        # P6: (A10-A00)(B00+B01)
        # P7: (A01-A11)(B10+B11)
        
        # U, V entries for each product
        # P1
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        # W entries for P1: contributes to C00 (1), C11 (1)
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # P2
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        # W entries for P2: contributes to C10 (1), C11 (-1)
        W[(k0+0)*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # P3
        U[(i0+0)*m + (j0+0)][col] += 1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += -1
        # W entries for P3: contributes to C01 (1), C11 (1)
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # P4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+0)][col] += -1
        # W entries for P4: contributes to C00 (1), C10 (1)
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        col += 1
        
        # P5
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        # W entries for P5: contributes to C00 (-1), C01 (1)
        W[(k0+0)*n + (i0+0)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        col += 1
        
        # P6
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+0)][col] += -1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        # W entries for P6: contributes to C11 (1)
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # P7
        U[(i0+0)*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        # W entries for P7: contributes to C00 (1)
        W[(k0+0)*n + (i0+0)][col] += 1
        col += 1

    def add_schoolbook(i0, j0, k0, ni, nj, nk):
        nonlocal col
        for ai in range(ni):
            for aj in range(nj):
                for ak in range(nk):
                    u_row = (i0 + ai) * m + (j0 + aj)
                    v_row = (j0 + aj) * p + (k0 + ak)
                    w_row = (k0 + ak) * n + (i0 + ai)
                    U[u_row][col] = 1
                    V[v_row][col] = 1
                    W[w_row][col] = 1
                    col += 1

    # Process blocks
    # i=0-1, j=0-1, k=0-1
    add_strassen_222(0, 0, 0)
    # i=0-1, j=0-1, k=2-3
    add_strassen_222(0, 0, 2)
    # i=0-1, j=0-1, k=4-5
    add_strassen_222(0, 0, 4)
    # i=0-1, j=2-3, k=0-1
    add_strassen_222(0, 2, 0)
    # i=0-1, j=2-3, k=2-3
    add_strassen_222(0, 2, 2)
    # i=0-1, j=2-3, k=4-5
    add_strassen_222(0, 2, 4)
    
    # i=0-1, j=4, k=0-1 (2x1x2)
    add_schoolbook(0, 4, 0, 2, 1, 2)
    # i=0-1, j=4, k=2-3
    add_schoolbook(0, 4, 2, 2, 1, 2)
    # i=0-1, j=4, k=4-5
    add_schoolbook(0, 4, 4, 2, 1, 2)
    
    # i=2, j=0-1, k=0-1 (1x2x2)
    add_schoolbook(2, 0, 0, 1, 2, 2)
    # i=2, j=0-1, k=2-3
    add_schoolbook(2, 0, 2, 1, 2, 2)
    # i=2, j=0-1, k=4-5
    add_schoolbook(2, 0, 4, 1, 2, 2)
    
    # i=2, j=2-3, k=0-1
    add_schoolbook(2, 2, 0, 1, 2, 2)
    # i=2, j=2-3, k=2-3
    add_schoolbook(2, 2, 2, 1, 2, 2)
    # i=2, j=2-3, k=4-5
    add_schoolbook(2, 2, 4, 1, 2, 2)
    
    # i=2, j=4, k=0-1 (1x1x2)
    add_schoolbook(2, 4, 0, 1, 1, 2)
    # i=2, j=4, k=2-3
    add_schoolbook(2, 4, 2, 1, 1, 2)
    # i=2, j=4, k=4-5
    add_schoolbook(2, 4, 4, 1, 1, 2)

    return {"U": U, "V": V, "W": W}
