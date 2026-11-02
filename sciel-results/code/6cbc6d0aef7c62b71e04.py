import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products definition
    # Each product is defined by:
    # a_coeffs: dict mapping (i, j) to coeff for A[i, j]
    # b_coeffs: dict mapping (j, k) to coeff for B[j, k]
    # c_coeffs: dict mapping (i, k) to coeff for C[i, k]
    
    # We need to map local (i, j) and (j, k) to global indices.
    # Block 1: A uses j in [0, 1], B uses j in [0, 1].
    # Block 2: A uses j in [2, 3], B uses j in [2, 3].
    
    # Strassen 7 products for 2x2x2
    # Local indices: a in {0,1}, b in {0,1}, c in {0,1}
    
    strassen_products = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M0 = (a00+a11)(b00+b11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}), # Wait, reconstruction is separate.
         # Actually, M0 contributes to C00 and C11.
         # C00 += M0, C11 += M0.
         # So c_coeffs for M0 in the sum is {(0,0): 1, (1,1): 1}
    ]
    
    # Let's define the full list of 7 products with their contributions to C
    # M0: a=(0,0:1, 1,1:1), b=(0,0:1, 1,1:1), c=(0,0:1, 1,1:1)
    # M1: a=(1,0:1, 1,1:1), b=(0,0:1), c=(1,0:1, 1,1:-1)
    # M2: a=(0,0:1), b=(0,1:1, 1,1:-1), c=(0,1:1, 1,1:1)
    # M3: a=(1,1:1), b=(1,0:1, 0,0:-1), c=(0,0:1, 1,0:1)
    # M4: a=(0,0:1, 0,1:1), b=(1,1:1), c=(0,0:-1, 0,1:1)
    # M5: a=(1,0:1, 0,0:-1), b=(0,0:1, 0,1:1), c=(1,1:1)
    # M6: a=(0,1:1, 1,1:-1), b=(1,0:1, 1,1:1), c=(0,0:1)
    
    products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col = 0
    for block in [0, 1]:
        # block 0: j offset 0
        # block 1: j offset 2
        j_offset = block * 2
        
        # B has 4 columns. We split into two 2-column blocks?
        # No, B is 2x4 in the block.
        # The Strassen algorithm is for 2x2x2.
        # So we need to apply it to each pair of columns of the 2x4 B block.
        # B_T is 2x4. Columns 0,1 and 2,3.
        
        for b_col_block in [0, 1]:
            k_offset = b_col_block * 2
            
            for a_coeffs, b_coeffs, c_coeffs in products:
                # Update U
                for (li, lj), val in a_coeffs.items():
                    global_i = li
                    global_j = j_offset + lj
                    U[global_i * m + global_j][col] += val
                
                # Update V
                for (lj, lk), val in b_coeffs.items():
                    global_j = j_offset + lj
                    global_k = k_offset + lk
                    V[global_j * p + global_k][col] += val
                
                # Update W
                for (li, lk), val in c_coeffs.items():
                    global_i = li
                    global_k = k_offset + lk
                    W[global_k * n + global_i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
