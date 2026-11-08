"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2
    # Local indices: A: 0,1,2,3; B: 0,1,2,3; C: 0,1,2,3
    U_strassen = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    V_strassen = [
        [1, 0, 0, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1], # m3
        [0, 0, 1, -1], # m4
        [0, 0, 0, 1],  # m5
        [1, 1, 0, 0],  # m6
        [0, 0, 1, 1]   # m7
    ]
    W_strassen = [
        [1, 0, 0, 1],  # m1 -> C00, C11
        [0, 0, 1, 0],  # m2 -> C10
        [0, 1, 0, 1],  # m3 -> C01, C11
        [1, 0, 1, 0],  # m4 -> C00, C10
        [-1, 0, 0, 0], # m5 -> C00
        [0, 0, 0, 1],  # m6 -> C11
        [1, 0, 0, 0]   # m7 -> C00
    ]

    def add_block(base_rank, a_indices, b_indices, c_indices):
        """
        Adds a 2x2x2 Strassen block to the global decomposition.
        a_indices: list of 4 global indices for A
        b_indices: list of 4 global indices for B
        c_indices: list of 4 global indices for C
        """
        for k in range(7):
            r = base_rank + k
            for local_idx, global_idx in enumerate(a_indices):
                U[global_idx][r] = U_strassen[k][local_idx]
            for local_idx, global_idx in enumerate(b_indices):
                V[global_idx][r] = V_strassen[k][local_idx]
            for local_idx, global_idx in enumerate(c_indices):
                W[global_idx][r] = W_strassen[k][local_idx]

    # Part 1: C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2:4] B[2:4, 0:2]
    # A indices: 0,1,2,3 (top-left 2x2 of A) and 4,5,6,7 (top-right 2x2 of A)
    # Note: A is 2x4. 
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # Wait, standard row-major indexing for A (2x4):
    # 0 1 2 3
    # 4 5 6 7
    
    # Block A_L (cols 0,1): indices 0,1,4,5
    # Block A_R (cols 2,3): indices 2,3,6,7
    
    # B is 4x4.
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    
    # Block B_LL (rows 0,1 cols 0,1): indices 0,1,4,5
    # Block B_LR (rows 0,1 cols 2,3): indices 2,3,6,7
    # Block B_RL (rows 2,3 cols 0,1): indices 8,9,12,13
    # Block B_RR (rows 2,3 cols 2,3): indices 10,11,14,15
    
    # C is 2x4.
    # 0 1 2 3
    # 4 5 6 7
    
    # C_L (cols 0,1): indices 0,1,4,5
    # C_R (cols 2,3): indices 2,3,6,7

    # Part 1, Prod 1: A_L B_LL
    a_indices_1 = [0, 1, 4, 5]
    b_indices_1 = [0, 1, 4, 5]
    c_indices_1 = [0, 1, 4, 5]
    add_block(0, a_indices_1, b_indices_1, c_indices_1)

    # Part 1, Prod 2: A_R B_RL
    a_indices_2 = [2, 3, 6, 7]
    b_indices_2 = [8, 9, 12, 13]
    c_indices_2 = [0, 1, 4, 5] # Output to C_L
    add_block(7, a_indices_2, b_indices_2, c_indices_2)

    # Part 2, Prod 1: A_L B_LR
    a_indices_3 = [0, 1, 4, 5]
    b_indices_3 = [2, 3, 6, 7]
    c_indices_3 = [2, 3, 6, 7] # Output to C_R
    add_block(14, a_indices_3, b_indices_3, c_indices_3)

    # Part 2, Prod 2: A_R B_RR
    a_indices_4 = [2, 3, 6, 7]
    b_indices_4 = [10, 11, 14, 15]
    c_indices_4 = [2, 3, 6, 7] # Output to C_R
    add_block(21, a_indices_4, b_indices_4, c_indices_4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
