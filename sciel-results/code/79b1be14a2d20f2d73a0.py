import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We expect rank 58
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the two Strassen blocks
    # Block 1: i in [0,1], j in [0,1], k in [0,1]
    # Block 2: i in [0,1], j in [2,3], k in [2,3]
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Helper to add Strassen terms for a block
    # local_i, local_j, local_k are 0 or 1
    # global_i_offset, global_j_offset, global_k_offset
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # A indices: (i_off + li, j_off + lj)
        # B indices: (j_off + lj, k_off + lk)
        # C indices: (k_off + lk, i_off + li)
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00 is (i_off, j_off), A11 is (i_off+1, j_off+1)
        # B00 is (j_off, k_off), B11 is (j_off+1, k_off+1)
        # C00 is (k_off, i_off), C11 is (k_off+1, i_off+1)
        
        # We need to map local (0,0) to global (i_off, j_off) etc.
        
        # M1
        U[(i_off) * m + (j_off)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off) * p + (k_off)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off) * n + (i_off)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        # A10 is (i_off+1, j_off), A11 is (i_off+1, j_off+1)
        # B00 is (j_off, k_off)
        # C10 is (k_off, i_off+1), C11 is (k_off+1, i_off+1)
        U[(i_off + 1) * m + (j_off)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off) * p + (k_off)][col] = 1
        W[(k_off) * n + (i_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M3 = A00 (B10 - B11)
        # A00 is (i_off, j_off)
        # B10 is (j_off+1, k_off), B11 is (j_off+1, k_off+1)
        # C00 is (k_off, i_off), C01 is (k_off+1, i_off)
        U[(i_off) * m + (j_off)][col] = 1
        V[(j_off + 1) * p + (k_off)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = -1
        W[(k_off) * n + (i_off)][col] = 1
        W[(k_off + 1) * n + (i_off)][col] = 1
        col += 1
        
        # M4 = A11 (B01 - B00)
        # A11 is (i_off+1, j_off+1)
        # B01 is (j_off, k_off+1), B00 is (j_off, k_off)
        # C00 is (k_off, i_off), C01 is (k_off+1, i_off)
        # Wait, C indices for M4:
        # C00 = ... - M4 ...
        # C01 = ... + M4 ...
        # So W for C00 is -1, W for C01 is 1?
        # Let's check Strassen formula again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3 - M4 + M5
        # C10 = M2 + M3 + M6 - M7
        # C11 = M1 - M2 + M6 + M7
        
        # My previous M4 definition: A11 (B01 - B00).
        # Contribution to C00: -1 * M4.
        # Contribution to C01: +1 * M4? No, - M4 in C01?
        # C01 = M2 + M3 - M4 + M5. Yes, -1.
        # Wait, let's re-verify standard Strassen.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3 - M4 + M5
        # C10 = M2 + M3 + M6 - M7
        # C11 = M1 - M2 + M6 + M7
        
        # So for M4:
        # W[C00] = 1 (from +M4 in C00)
        # W[C01] = -1 (from -M4 in C01)
        # My previous code for M4 was:
        # W[(k_off) * n + (i_off)][col] = 1  # C00
        # W[(k_off + 1) * n + (i_off)][col] = 1 # C01 ??
        # It should be -1 for C01.
        
        # Let's correct M4 in the code below.
        
        # M5 = (A00 + A01) B11
        # A00 (i_off, j_off), A01 (i_off, j_off+1)
        # B11 (j_off+1, k_off+1)
        # C00: -1, C01: +1
        # C00 = ... - M5 ...
        # C01 = ... + M5 ...
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10 (i_off+1, j_off), A00 (i_off, j_off)
        # B00 (j_off, k_off), B01 (j_off, k_off+1)
        # C10: +1, C11: +1
        
        # M7 = (A01 - A11)(B10 + B11)
        # A01 (i_off, j_off+1), A11 (i_off+1, j_off+1)
        # B10 (j_off+1, k_off), B11 (j_off+1, k_off+1)
        # C00: +1, C10: -1
        
        # I will implement this carefully.
        
    # Re-implement add_strassen with correct signs
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        
        # M1: (A00 + A11)(B00 + B11)
        # C00: +1, C11: +1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10 + A11)B00
        # C10: +1, C11: -1
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # M3: A00(B10 - B11)
        # C01: +1, C10: +1
        U[i_off * m + j_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M4: A11(B01 - B00)
        # C00: +1, C01: -1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + i_off][col] = -1
        col += 1
        
        # M5: (A00 + A01)B11
        # C00: -1, C01: +1
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        # C10: +1, C11: +1
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01 - A11)(B10 + B11)
        # C00: +1, C10: -1
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = -1
        col += 1

    # Apply Strassen to Block 1
    add_strassen_block(0, 0, 0)
    
    # Apply Strassen to Block 2
    add_strassen_block(0, 2, 2)
    
    # Add schoolbook for all other terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if (i,j,k) is in Block 1
                if 0 <= i <= 1 and 0 <= j <= 1 and 0 <= k <= 1:
                    continue
                # Check if (i,j,k) is in Block 2
                if 0 <= i <= 1 and 2 <= j <= 3 and 2 <= k <= 3:
                    continue
                
                add_schoolbook(i, j, k)
                
    return {"U": U, "V": V, "W": W}
