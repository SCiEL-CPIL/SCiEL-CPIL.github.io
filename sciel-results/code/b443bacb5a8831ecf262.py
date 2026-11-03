"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    R = 29
    
    # Initialize U, V, W
    U = [[0.0] * R for _ in range(n)]
    V = [[0.0] * R for _ in range(m)]
    W = [[0.0] * R for _ in range(p)]
    
    # Strassen factors for 2x2x2
    # U_S: 2x7, V_S: 2x7, W_S: 2x7
    U_S = [
        [1, 0, 1, 0, 1, -1, 1], # i=0 (A11, A12)
        [1, 1, 0, 1, 0, 1, -1]  # i=1 (A21, A22)
    ]
    V_S = [
        [1, 1, 1, 1, 0, 1, 1], # j=0 (B11, B12)
        [1, 0, -1, -1, 1, 0, 1] # j=1 (B21, B22)
    ]
    W_S = [
        [1, 1, 0, 1, -1, 0, 1], # k=0 (C11, C21)
        [1, -1, 1, 0, 1, 1, 0]  # k=1 (C12, C22)
    ]
    
    # We have 4 blocks of 2x2x2
    # Block 0: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 1: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 2: i in {0,1}, j in {0,1}, k in {4,5}
    # Block 3: i in {0,1}, j in {0,1}, k in {6,7}
    
    for b in range(4):
        offset = b * 7
        k_offset = b * 2
        for i in range(2):
            for c in range(7):
                U[i][offset + c] += U_S[i][c]
        for j in range(2):
            for c in range(7):
                V[j][offset + c] += V_S[j][c]
        for k in range(2):
            for c in range(7):
                W[k_offset + k][offset + c] += W_S[k][c]
                
    # Part 2: 2x1x8 part (j=2)
    # T_{i, 2, k} = 1
    # Use rank 1
    offset = 28
    for i in range(2):
        U[i][offset] = 1.0
    V[2][offset] = 1.0
    for k in range(8):
        W[k][offset] = 1.0
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
