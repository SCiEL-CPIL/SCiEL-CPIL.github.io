"""Matrix multiplication 2x2 x 2x13 - tensor decomposition search."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen for 2x2x2 blocks and schoolbook for the remaining 2x2x1
    # 6 blocks of 2x2x2 (rank 7 each) + 1 block of 2x2x1 (rank 4)
    # Total rank = 6 * 7 + 4 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for a 2x2x2 block
    # U coeffs (A): [a, b, c, d]
    # V coeffs (B): [e, f, g, h]
    # W coeffs (C): [c00, c10, c01, c11]
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1: (a+d)(e+h) -> C00+1, C11+1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2: (c+d)e   -> C10+1, C11-1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3: a(f-h)    -> C01+1, C11+1
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]), # M4: d(g-e)    -> C00+1, C10+1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5: (a+b)h    -> C00-1, C01+1
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6: (c-a)(e+f)-> C11+1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7: (b-d)(g+h)-> C00+1
    ]
    
    # Process 6 blocks of 2 columns
    for block_idx in range(6):
        k_start = block_idx * 2
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U
            for idx, val in enumerate(u_coeffs):
                U[idx][col] = val
            
            # Fill V
            # e=B[0,k], f=B[0,k+1], g=B[1,k], h=B[1,k+1]
            # V index: j*p + k
            # e: 0*p + k = k
            # f: 0*p + k+1 = k+1
            # g: 1*p + k = p+k
            # h: 1*p + k+1 = p+k+1
            v_indices = [k_start, k_start + 1, p + k_start, p + k_start + 1]
            for idx, val in enumerate(v_coeffs):
                V[v_indices[idx]][col] = val
            
            # Fill W
            # c00=C[0,k], c10=C[1,k], c01=C[0,k+1], c11=C[1,k+1]
            # W index: k*n + i
            # c00: k*n + 0 = 2k
            # c10: k*n + 1 = 2k+1
            # c01: (k+1)*n + 0 = 2k+2
            # c11: (k+1)*n + 1 = 2k+3
            w_indices = [2 * k_start, 2 * k_start + 1, 2 * k_start + 2, 2 * k_start + 3]
            for idx, val in enumerate(w_coeffs):
                W[w_indices[idx]][col] = val
            
            col += 1
            
    # Process the last single column (k=12)
    k_last = 12
    # Schoolbook for 2x2x1
    # C[i, k] = A[i, 0] B[0, k] + A[i, 1] B[1, k]
    
    # Term 1: A[0,0] B[0,12] -> C[0,12]
    U[0][col] = 1
    V[k_last][col] = 1 # B[0,12] is at index 0*13 + 12 = 12
    W[2 * k_last][col] = 1 # C[0,12] is at index 12*2 + 0 = 24
    col += 1
    
    # Term 2: A[0,1] B[1,12] -> C[0,12]
    U[1][col] = 1
    V[p + k_last][col] = 1 # B[1,12] is at index 1*13 + 12 = 25
    W[2 * k_last][col] = 1 # C[0,12]
    col += 1
    
    # Term 3: A[1,0] B[0,12] -> C[1,12]
    U[2][col] = 1
    V[k_last][col] = 1 # B[0,12]
    W[2 * k_last + 1][col] = 1 # C[1,12] is at index 12*2 + 1 = 25
    col += 1
    
    # Term 4: A[1,1] B[1,12] -> C[1,12]
    U[3][col] = 1
    V[p + k_last][col] = 1 # B[1,12]
    W[2 * k_last + 1][col] = 1 # C[1,12]
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
