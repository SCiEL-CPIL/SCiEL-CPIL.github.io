import time

def construct():
    n, m, p = 3, 4, 6
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    def add_strassen_222(i_base, j_base, k_base):
        """Add 7 Strassen terms for a 2x2x2 block."""
        nonlocal col
        
        # A indices: (i_base, j_base), (i_base, j_base+1), (i_base+1, j_base), (i_base+1, j_base+1)
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        # B indices: (j_base, k_base), (j_base, k_base+1), (j_base+1, k_base), (j_base+1, k_base+1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # C/W indices: (k_base, i_base), (k_base+1, i_base), (k_base, i_base+1), (k_base+1, i_base+1)
        # W index = k*n + i
        w00 = k_base * n + i_base
        w01 = (k_base + 1) * n + i_base
        w10 = k_base * n + (i_base + 1)
        w11 = (k_base + 1) * n + (i_base + 1)
        
        # M1 = (a+d)(e+h) -> C11 += 1, C22 += 1
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        
        # M2 = (c+d)e -> C21 += 1, C22 += -1
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[w10][col] = 1; W[w11][col] = -1
        col += 1
        
        # M3 = a(f-h) -> C12 += 1, C22 += 1
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[w01][col] = 1; W[w11][col] = 1
        col += 1
        
        # M4 = d(g-e) -> C11 += 1, C21 += 1
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        
        # M5 = (a+b)h -> C11 += -1, C12 += 1
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C22 += 1
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C11 += 1
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[w00][col] = 1
        col += 1
    
    def add_strassen_223(i_base, j_base, k_base):
        """Add 9 terms for a 2x2x3 block: 7 Strassen for 2x2x2 + 2 schoolbook for 2x2x1."""
        nonlocal col
        
        # First, handle the 2x2x2 part (columns k_base, k_base+1)
        add_strassen_222(i_base, j_base, k_base)
        
        # Then handle the 2x2x1 part (column k_base+2) with schoolbook
        # A[i_base, j_base] * B[j_base, k_base+2] -> C[i_base, k_base+2]
        # A[i_base, j_base+1] * B[j_base+1, k_base+2] -> C[i_base, k_base+2]
        # A[i_base+1, j_base] * B[j_base, k_base+2] -> C[i_base+1, k_base+2]
        # A[i_base+1, j_base+1] * B[j_base+1, k_base+2] -> C[i_base+1, k_base+2]
        # That's 4 schoolbook terms, not 2! Let me reconsider.
        
        # Actually for 2x2x1, we need 4 terms (2 rows of A × 2 cols of A × 1 col of B)
        # So 2x2x3 = 7 + 4 = 11 terms, not 9.
        
        # Let me fix this: add 4 schoolbook terms for the remaining column
        for i_off in range(2):
            for j_off in range(2):
                i = i_base + i_off
                j = j_base + j_off
                k = k_base + 2
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 1: Row 0 of A: 1x4x6, schoolbook = 24 terms
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Part 2: Rows 1-2 of A: 2x4x6
    # Split into 2 j-groups of 2, and 2 k-groups of 3
    # Each 2x2x3 block uses 11 terms (7 Strassen + 4 schoolbook)
    # Total: 2 * 2 * 11 = 44 terms... that's worse than 42!
    
    # Let me reconsider. The original 42-term approach:
    # 2 j-groups × 3 k-groups of 2 = 6 blocks × 7 = 42 terms
    
    # Alternative: Can we do better?
    # What if we use a different decomposition for 2x4x6?
    
    # Actually, let me just go with the proven 66-term solution but verify it's correct.
    # The key is to make sure the decomposition is exact.
    
    # Let me reset and use the clean 66-term approach that we know works.
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Row 0: schoolbook 24 terms
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: 6 Strassen blocks of 2x2x2
    for j_block in range(2):
        for k_block in range(3):
            j_base = j_block * 2
            k_base = k_block * 2
            add_strassen_222(1, j_base, k_base)
    
    return {"U": U, "V": V, "W": W}
