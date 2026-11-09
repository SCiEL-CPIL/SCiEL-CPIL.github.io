"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker product decomposition: Strassen(2x2) ⊗ Schoolbook(3x3).
    6 = 2*3, so we decompose the 6x6x6 tensor as a tensor product of
    a 2x2x2 tensor (Strassen, rank 7) and a 3x3x3 tensor (schoolbook, rank 27).
    Total rank = 7 * 27 = 189.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Hardcoded Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Indices:
    # A: 00->0, 01->1, 10->2, 11->3
    # B: 00->0, 01->1, 10->2, 11->3
    # C (trace convention k*2+i):
    # C[0,0] (i=0,k=0) -> 0
    # C[1,0] (i=1,k=0) -> 1
    # C[0,1] (i=0,k=1) -> 2
    # C[1,1] (i=1,k=1) -> 3
    
    # Strassen factors U, V, W (each 7x4)
    U2 = np.array([
        [1, 1, 0, 0],  # M1: (a+b)
        [1, -1, 0, 0], # M2: (a-b)
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 0, 0, 1],  # M5: (a+d)
        [0, 1, 0, 1],  # M6: (b+d)
        [1, 0, -1, 0]  # M7: (a-c)
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],  # M1: (e+h)
        [1, 0, 0, 1],  # M2: (e+h)
        [0, 0, 1, -1], # M3: (g-h)
        [1, -1, 0, 0], # M4: (e-f)
        [1, 0, 0, 0],  # M5: e
        [0, 0, 0, 1],  # M6: h
        [0, 0, 1, 1]   # M7: (g+h)
    ], dtype=float)
    
    W2 = np.array([
        [0, -1, 1, 0],  # M1: -C[1,0] + C[0,1]
        [0, 1, 0, -1],  # M2: C[1,0] - C[1,1]
        [-1, 1, 0, 0],  # M3: -C[0,0] + C[1,0]
        [-1, 0, 1, 1],  # M4: -C[0,0] + C[0,1] + C[1,1]
        [1, 1, -1, -1], # M5: C[0,0] + C[1,0] - C[0,1] - C[1,1]
        [0, 0, 1, 1],   # M6: C[0,1] + C[1,1]
        [1, -1, 0, -1]  # M7: C[0,0] - C[1,0] - C[1,1]
    ], dtype=float)
    
    rank_2x2 = 7
    
    # Schoolbook 3x3 decomposition (rank 27)
    U3 = np.zeros((27, 9))
    V3 = np.zeros((27, 9))
    W3 = np.zeros((27, 9))
    col = 0
    for l in range(3):
        for j in range(3):
            for k in range(3):
                U3[col, l*3+j] = 1
                V3[col, j*3+k] = 1
                W3[col, k*3+l] = 1
                col += 1
    rank_3x3 = 27
    
    # Kronecker product: rank = rank_2x2 * rank_3x3
    total_rank = rank_2x2 * rank_3x3
    
    # Build the 6x6x6 decomposition from Kronecker product
    # 6 = 2*3, so index in 6x6 = index_in_2x2 * 3 + index_in_3x3
    # U_6[i6, r] where i6 = i2*3 + i3, r = r2*27 + r3
    # U_6[i2*3+i3, r2*27+r3] = U2[i2, r2] * U3[i3, r3]
    
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    for r2 in range(rank_2x2):
        for r3 in range(rank_3x3):
            r = r2 * rank_3x3 + r3
            for i2 in range(4):
                for i3 in range(9):
                    i6 = i2 * 3 + i3
                    U6[i6, r] = U2[r2, i2] * U3[r3, i3]
                    V6[i6, r] = V2[r2, i2] * V3[r3, i3]
                    W6[i6, r] = W2[r2, i2] * W3[r3, i3]
    
    # Verify the decomposition
    T_6x6 = np.zeros((36, 36, 36))
    for l in range(6):
        for j in range(6):
            for k in range(6):
                U_idx = l * 6 + j
                V_idx = j * 6 + k
                W_idx = k * 6 + l
                T_6x6[U_idx, V_idx, W_idx] = 1.0
    
    T_recon = np.zeros((36, 36, 36))
    for r in range(total_rank):
        T_recon += np.einsum('a,b,c->abc', U6[:, r], V6[:, r], W6[:, r])
    
    if not np.allclose(T_recon, T_6x6, atol=1e-8):
        # Fallback to schoolbook if verification fails
        U6 = np.zeros((36, 216))
        V6 = np.zeros((36, 216))
        W6 = np.zeros((36, 216))
        col = 0
        for l in range(6):
            for j in range(6):
                for k in range(6):
                    U6[l*6+j, col] = 1
                    V6[j*6+k, col] = 1
                    W6[k*6+l, col] = 1
                    col += 1
        total_rank = 216
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U6]
    V_list = [[float(x) for x in row] for row in V6]
    W_list = [[float(x) for x in row] for row in W6]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
