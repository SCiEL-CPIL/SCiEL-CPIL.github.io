"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Strategy: split p=6 into p1=4 and p2=2
    # Part 1: 2x4x4 multiplication (split m into two 2s, each 2x2x4 = two 2x2x2 Strassen)
    # Part 2: 2x4x2 multiplication (trivial rank 8)
    # Total rank: 2*(2*7) + 8 = 28 + 8 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 24 x 36
    W = [[0] * rank for _ in range(p * n)]   # 12 x 36
    
    # Strassen coefficients for 2x2x2
    strassen_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}),         # M2
        ({0: 1}, {1: 1, 3: -1}),        # M3
        ({3: 1}, {2: 1, 0: -1}),        # M4
        ({0: 1, 1: 1}, {3: 1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),  # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1})   # M7
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1}              # M7
    ]
    
    col = 0
    # Part 1: 2x4x4 (k=0..3)
    # Split m=4 into two groups of 2: s=0 (j=0,1), s=1 (j=2,3)
    # For each s, split k=4 into two groups of 2: q=0 (k=0,1), q=1 (k=2,3)
    # Each is a 2x2x2 Strassen
    
    for s in range(2):  # which half of m (j dimension)
        for q in range(2):  # which half of k (first 4 cols of output)
            for m_idx in range(7):
                u_coeffs = strassen_steps[m_idx][0]
                v_coeffs = strassen_steps[m_idx][1]
                w_coeffs = W_coeffs[m_idx]
                
                # Update U: A[i, j] where i=0,1; j=2*s+{0,1}
                for la, coeff in u_coeffs.items():
                    i = la // 2
                    j_local = la % 2
                    j_global = 2 * s + j_local
                    u_row = i * m + j_global
                    U[u_row][col] += coeff
                
                # Update V: B[j, k] where j=2*s+{0,1}; k=2*q+{0,1}
                for lb, coeff in v_coeffs.items():
                    i_local = lb // 2
                    j_local = lb % 2
                    j_global = 2 * s + i_local
                    k_global = 2 * q + j_local
                    v_row = j_global * p + k_global
                    V[v_row][col] += coeff
                
                # Update W: C[k, i] where k=2*q+{0,1}; i=0,1
                for lc, coeff in w_coeffs.items():
                    i = lc // 2
                    k_local = lc % 2
                    k_global = 2 * q + k_local
                    w_row = k_global * n + i
                    W[w_row][col] += coeff
                
                col += 1
    
    # Part 2: 2x4x2 (k=4,5)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Trivial: 8 terms (i=0,1; j=0,1,2,3; k=4,5)
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
