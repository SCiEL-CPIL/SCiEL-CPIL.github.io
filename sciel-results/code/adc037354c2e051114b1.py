"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # P0 = (a00+a11)(b00+b11)
    # P1 = (a10+a11)b00
    # P2 = a00(b10-b11)
    # P3 = a11(b01-b00)
    # P4 = (a00+a01)b11
    # P5 = (a10-a00)(b00+b01)
    # P6 = (a01-a11)(b10+b11)
    
    # C00 = P0 + P3 - P5 - P6
    # C01 = P2 + P6
    # C10 = P1 + P5
    # C11 = P0 - P1 + P2 + P4
    
    # U vectors for A (4 elements)
    U_strassen = [
        [1, 0, 0, 1], # P0
        [0, 0, 1, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 0, 0, 1], # P3
        [1, 1, 0, 0], # P4
        [-1, 0, 1, 0],# P5
        [0, 1, 0, -1] # P6
    ]
    
    # V vectors for B (4 elements)
    V_strassen = [
        [1, 0, 0, 1], # P0
        [1, 0, 0, 0], # P1
        [0, 0, 1, -1],# P2
        [0, 1, -1, 0],# P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, 0, 1, 1]  # P6
    ]
    
    # W vectors for C (4 elements)
    # C00, C01, C10, C11
    W_strassen = [
        [1, 0, 0, 0], # P0 -> C00, C11
        [0, 0, 1, 0], # P1 -> C10, C11
        [0, 1, 0, 1], # P2 -> C01, C11
        [1, 0, 0, 0], # P3 -> C00
        [0, 0, 0, 1], # P4 -> C11
        [0, 0, 1, 0], # P5 -> C10, C00
        [0, 1, 0, 0]  # P6 -> C01, C00
    ]
    # Wait, W maps P to C.
    # C = sum P_k * W_k
    # C00 = 1*P0 + 1*P3 - 1*P5 - 1*P6
    # C01 = 1*P2 + 1*P6
    # C10 = 1*P1 + 1*P5
    # C11 = 1*P0 - 1*P1 + 1*P2 + 1*P4
    
    # So W matrix (4x7) where W[c_idx, p_idx]
    W_strassen = [
        [1, 0, 0, 1, 0, -1, -1], # C00
        [0, 0, 1, 0, 0, 0, 1],   # C01
        [0, 1, 0, 0, 0, 1, 0],   # C10
        [1, -1, 1, 0, 1, 0, 0]   # C11
    ]
    
    col = 0
    # Loop over the two j-blocks (0, 1)
    for j_block in range(2):
        # j indices in this block: 2*j_block, 2*j_block+1
        j0 = 2 * j_block
        j1 = 2 * j_block + 1
        
        # Loop over the three k-pairs (0, 1, 2)
        for k_pair in range(3):
            # k indices in this pair: 2*k_pair, 2*k_pair+1
            k0 = 2 * k_pair
            k1 = 2 * k_pair + 1
            
            # Loop over the 7 Strassen terms
            for s in range(7):
                # Update U
                # A indices: i, j
                # U_strassen[s] has 4 entries for a00, a01, a10, a11
                # a00 corresponds to i=0, j=j0
                # a01 corresponds to i=0, j=j1
                # a10 corresponds to i=1, j=j0
                # a11 corresponds to i=1, j=j1
                
                U[0 * m + j0][col] += U_strassen[s][0]
                U[0 * m + j1][col] += U_strassen[s][1]
                U[1 * m + j0][col] += U_strassen[s][2]
                U[1 * m + j1][col] += U_strassen[s][3]
                
                # Update V
                # B indices: j, k
                # V_strassen[s] has 4 entries for b00, b01, b10, b11
                # b00 corresponds to j=j0, k=k0
                # b01 corresponds to j=j0, k=k1
                # b10 corresponds to j=j1, k=k0
                # b11 corresponds to j=j1, k=k1
                
                V[j0 * p + k0][col] += V_strassen[s][0]
                V[j0 * p + k1][col] += V_strassen[s][1]
                V[j1 * p + k0][col] += V_strassen[s][2]
                V[j1 * p + k1][col] += V_strassen[s][3]
                
                # Update W
                # C indices: k, i
                # W_strassen[s] has 4 entries for c00, c01, c10, c11
                # c00 corresponds to k=k0, i=0 -> index k0*n + 0
                # c01 corresponds to k=k1, i=0 -> index k1*n + 0
                # c10 corresponds to k=k0, i=1 -> index k0*n + 1
                # c11 corresponds to k=k1, i=1 -> index k1*n + 1
                
                W[k0 * n + 0][col] += W_strassen[0][s]
                W[k1 * n + 0][col] += W_strassen[1][s]
                W[k0 * n + 1][col] += W_strassen[2][s]
                W[k1 * n + 1][col] += W_strassen[3][s]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
