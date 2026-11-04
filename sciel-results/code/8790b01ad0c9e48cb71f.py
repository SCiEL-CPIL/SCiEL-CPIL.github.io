import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # i_offset: global row offset for A (0 or 2)
    # j_offset: global col offset for A / row offset for B (0 or 2)
    # k_offset: global col offset for B / row offset for C (0 or 2)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices 0,1 map to global i_off+0, i_off+1 etc.
        # A indices: (i_off+li, j_off+lj)
        # B indices: (j_off+ljk, k_off+lk)
        # C indices: (i_off+li, k_off+lk)
        
        # Strassen formulas
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to add 7 columns.
        for r in range(7):
            c = col + r
            # Determine coefficients for U, V, W based on r
            # This is tedious to hardcode for each block, 
            # but the structure is the same.
            
            # Let's define the terms for each M
            # We will fill U, V, W for this specific block.
            
            # It is easier to generate the U, V, W entries for the 7 multiplications
            # and add them to the global arrays.
            
            pass # Placeholder

    # Instead of a complex helper, we can just loop through the 4 blocks
    # and the schoolbook part.
    
    # Block 1: i=0,1; j=0,1; k=0,1
    # Block 2: i=0,1; j=2,3; k=0,1
    # Block 3: i=0,1; j=0,1; k=2,3
    # Block 4: i=0,1; j=2,3; k=2,3
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    for i_off, j_off, k_off in blocks:
        # Add Strassen for 2x2x2
        # We need to map local (li, lj, lk) to global (i, j, k)
        # i = i_off + li
        # j = j_off + lj
        # k = k_off + lk
        
        # Strassen coefficients
        # M1: A00+A11, B00+B11 -> C00, C11
        # M2: A10+A11, B00 -> C10, C11
        # M3: A00, B10-B11 -> C01, C11
        # M4: A11, B01-B00 -> C00, C10, C11
        # M5: A00+A01, B11 -> C00, C01
        # M6: A10-A00, B00+B01 -> C11
        # M7: A01-A11, B10+B11 -> C00, C11
        
        # We will create 7 columns.
        for r in range(7):
            c = col + r
            # Clear/Init
            # U entries
            # V entries
            # W entries
            
            # This requires careful mapping.
            # Let's define the U, V, W contributions for each M
            # M1
            if r == 0:
                # U: A00, A11
                U[(i_off+0)*m + (j_off+0)][c] += 1
                U[(i_off+1)*m + (j_off+1)][c] += 1
                # V: B00, B11
                V[(j_off+0)*p + (k_off+0)][c] += 1
                V[(j_off+1)*p + (k_off+1)][c] += 1
                # W: C00, C11
                W[(k_off+0)*n + (i_off+0)][c] += 1
                W[(k_off+1)*n + (i_off+1)][c] += 1
            elif r == 1:
                # U: A10, A11
                U[(i_off+1)*m + (j_off+0)][c] += 1
                U[(i_off+1)*m + (j_off+1)][c] += 1
                # V: B00
                V[(j_off+0)*p + (k_off+0)][c] += 1
                # W: C10, C11
                W[(k_off+0)*n + (i_off+1)][c] += 1
                W[(k_off+1)*n + (i_off+1)][c] += 1
            elif r == 2:
                # U: A00
                U[(i_off+0)*m + (j_off+0)][c] += 1
                # V: B10, B11
                V[(j_off+1)*p + (k_off+0)][c] += 1
                V[(j_off+1)*p + (k_off+1)][c] -= 1
                # W: C01, C11
                W[(k_off+1)*n + (i_off+0)][c] += 1
                W[(k_off+1)*n + (i_off+1)][c] += 1
            elif r == 3:
                # U: A11
                U[(i_off+1)*m + (j_off+1)][c] += 1
                # V: B01, B00
                V[(j_off+0)*p + (k_off+1)][c] += 1
                V[(j_off+0)*p + (k_off+0)][c] -= 1
                # W: C00, C10, C11
                W[(k_off+0)*n + (i_off+0)][c] += 1
                W[(k_off+0)*n + (i_off+1)][c] += 1
                W[(k_off+1)*n + (i_off+1)][c] += 1
            elif r == 4:
                # U: A00, A01
                U[(i_off+0)*m + (j_off+0)][c] += 1
                U[(i_off+0)*m + (j_off+1)][c] += 1
                # V: B11
                V[(j_off+1)*p + (k_off+1)][c] += 1
                # W: C00, C01
                W[(k_off+0)*n + (i_off+0)][c] += 1
                W[(k_off+1)*n + (i_off+0)][c] += 1
            elif r == 5:
                # U: A10, A00
                U[(i_off+1)*m + (j_off+0)][c] += 1
                U[(i_off+0)*m + (j_off+0)][c] -= 1
                # V: B00, B01
                V[(j_off+0)*p + (k_off+0)][c] += 1
                V[(j_off+0)*p + (k_off+1)][c] += 1
                # W: C11
                W[(k_off+1)*n + (i_off+1)][c] += 1
            elif r == 6:
                # U: A01, A11
                U[(i_off+0)*m + (j_off+1)][c] += 1
                U[(i_off+1)*m + (j_off+1)][c] -= 1
                # V: B10, B11
                V[(j_off+1)*p + (k_off+0)][c] += 1
                V[(j_off+1)*p + (k_off+1)][c] += 1
                # W: C00, C11
                W[(k_off+0)*n + (i_off+0)][c] += 1
                W[(k_off+1)*n + (i_off+1)][c] += 1
                
            col += 1
            
    # Schoolbook for i=2
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
