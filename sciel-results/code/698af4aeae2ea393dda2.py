"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen decomposition on 2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    # M2 = (c+d)e   -> C10 += 1
    # M3 = a(f-h)   -> C01 += 1
    # M4 = d(g-e)   -> C00 += 1, C10 += 1
    # M5 = (a+b)h   -> C00 -= 1, C01 += 1
    # M6 = (c-a)(e+f) -> C11 += 1
    # M7 = (b-d)(g+h) -> C00 += 1
    
    # Block offsets
    # i_offset in {0} (since n=2, only one i-block of size 2)
    # j_offset in {0, 2} (size 2)
    # k_offset in {0, 2, 4, 6} (size 2)
    
    col = 0
    for i_off in [0]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4, 6]:
                # 7 Strassen multiplications
                # M1
                U[i_off * m + j_off][col] = 1
                U[(i_off + 1) * m + (j_off + 1)][col] = 1
                V[j_off * p + k_off][col] = 1
                V[(j_off + 1) * p + (k_off + 1)][col] = 1
                W[k_off * n + i_off][col] = 1
                W[(k_off + 1) * n + (i_off + 1)][col] = 1
                col += 1
                
                # M2
                U[(i_off + 1) * m + j_off][col] = 1
                U[(i_off + 1) * m + (j_off + 1)][col] = 1
                V[j_off * p + k_off][col] = 1
                W[k_off * n + (i_off + 1)][col] = 1
                col += 1
                
                # M3
                U[i_off * m + j_off][col] = 1
                V[j_off * p + (k_off + 1)][col] = 1
                V[(j_off + 1) * p + k_off][col] = -1
                W[(k_off + 1) * n + i_off][col] = 1
                col += 1
                
                # M4
                U[(i_off + 1) * m + (j_off + 1)][col] = 1
                V[(j_off + 1) * p + k_off][col] = 1
                V[j_off * p + k_off][col] = -1
                W[k_off * n + i_off][col] = 1
                W[k_off * n + (i_off + 1)][col] = 1
                col += 1
                
                # M5
                U[i_off * m + j_off][col] = 1
                U[i_off * m + (j_off + 1)][col] = 1
                V[(j_off + 1) * p + (k_off + 1)][col] = 1
                W[k_off * n + i_off][col] = -1
                W[(k_off + 1) * n + i_off][col] = 1
                col += 1
                
                # M6
                U[(i_off + 1) * m + j_off][col] = 1
                U[i_off * m + j_off][col] = -1
                V[j_off * p + k_off][col] = 1
                V[j_off * p + (k_off + 1)][col] = 1
                W[(k_off + 1) * n + (i_off + 1)][col] = 1
                col += 1
                
                # M7
                U[i_off * m + (j_off + 1)][col] = 1
                U[(i_off + 1) * m + (j_off + 1)][col] = -1
                V[(j_off + 1) * p + k_off][col] = 1
                V[(j_off + 1) * p + (k_off + 1)][col] = 1
                W[k_off * n + i_off][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
