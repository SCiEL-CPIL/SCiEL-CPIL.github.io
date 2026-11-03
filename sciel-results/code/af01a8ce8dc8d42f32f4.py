"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition using 2,2,1 block split with Strassen for 2x2x2 blocks.
    n, m, p = DIMENSIONS
    # Block sizes for indices 0..4
    # 0,1 -> Block 0 (size 2)
    # 2,3 -> Block 1 (size 2)
    # 4   -> Block 2 (size 1)
    block_sizes = [2, 2, 1]
    
    # Precompute local decompositions for all block size combinations
    # We need U, V, W for (sI, sJ, sK)
    # Types:
    # (2,2,2) -> Strassen (Rank 7)
    # (2,2,1), (2,1,2), (1,2,2) -> Schoolbook (Rank 4)
    # (2,1,1), (1,2,1), (1,1,2) -> Schoolbook (Rank 2)
    # (1,1,1) -> Schoolbook (Rank 1)
    
    def get_local_decomp(sI, sJ, sK):
        if sI == 2 and sJ == 2 and sK == 2:
            # Strassen 2x2x2
            # U[2*2, 7], V[2*2, 7], W[2*2, 7]
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # M1 = (A11+A22)(B11+B22)
            # U: A11, A22. V: B11, B22. W: C11, C22.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Col 0: M1
            U[0][0] = 1; U[3][0] = 1 # A11, A22
            V[0][0] = 1; V[3][0] = 1 # B11, B22
            W[0][0] = 1; W[3][0] = 1 # C11, C22
            
            # Col 1: M2 = (A21+A22)B11
            U[2][1] = 1; U[3][1] = 1 # A21, A22
            V[0][1] = 1 # B11
            W[2][1] = 1; W[3][1] = -1 # C21, C22
            
            # Col 2: M3 = A11(B12-B22)
            U[0][2] = 1 # A11
            V[1][2] = 1; V[3][2] = -1 # B12, B22
            W[1][2] = 1; W[3][2] = 1 # C12, C22
            
            # Col 3: M4 = A22(B21-B11)
            U[3][3] = 1 # A22
            V[2][3] = 1; V[0][3] = -1 # B21, B11
            W[0][3] = 1; W[2][3] = 1 # C11, C21
            
            # Col 4: M5 = (A11+A12)B22
            U[0][4] = 1; U[1][4] = 1 # A11, A12
            V[3][4] = 1 # B22
            W[0][4] = -1; W[1][4] = 1 # C11, C12
            
            # Col 5: M6 = (A21-A11)(B11+B12)
            U[2][5] = 1; U[0][5] = -1 # A21, A11
            V[0][5] = 1; V[1][5] = 1 # B11, B12
            W[3][5] = 1 # C22
            
            # Col 6: M7 = (A12-A22)(B21+B22)
            U[1][6] = 1; U[3][6] = -1 # A12, A22
            V[2][6] = 1; V[3][6] = 1 # B21, B22
            W[0][6] = 1 # C11
            
            return U, V, W
        
        else:
            # Schoolbook for sI x sJ x sK
            rank = sI * sJ * sK
            U = [[0]*rank for _ in range(sI * sJ)]
            V = [[0]*rank for _ in range(sJ * sK)]
            W = [[0]*rank for _ in range(sK * sI)]
            
            col = 0
            for u in range(sI):
                for v in range(sJ):
                    for w in range(sK):
                        U[u * sJ + v][col] = 1
                        V[v * sK + w][col] = 1
                        W[w * sI + u][col] = 1
                        col += 1
            return U, V, W

    # Map global index to block index and local index
    def get_block_info(idx):
        if idx < 2:
            return 0, idx
        elif idx < 4:
            return 1, idx - 2
        else:
            return 2, idx - 4

    # Total rank
    total_rank = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                sI = block_sizes[I]
                sJ = block_sizes[J]
                sK = block_sizes[K]
                U_loc, V_loc, W_loc = get_local_decomp(sI, sJ, sK)
                r = len(U_loc[0])
                total_rank += r

    # Initialize global matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    current_col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                sI = block_sizes[I]
                sJ = block_sizes[J]
                sK = block_sizes[K]
                U_loc, V_loc, W_loc = get_local_decomp(sI, sJ, sK)
                r = len(U_loc[0])

                # Fill global matrices
                for u in range(sI):
                    for v in range(sJ):
                        i = 2 * I + u if I < 2 else 4
                        j = 2 * J + v if J < 2 else 4
                        global_u_idx = i * m + j
                        for c in range(r):
                            U[global_u_idx][current_col + c] = U_loc[u * sJ + v][c]

                for v in range(sJ):
                    for w in range(sK):
                        j = 2 * J + v if J < 2 else 4
                        k = 2 * K + w if K < 2 else 4
                        global_v_idx = j * p + k
                        for c in range(r):
                            V[global_v_idx][current_col + c] = V_loc[v * sK + w][c]

                for w in range(sK):
                    for u in range(sI):
                        k = 2 * K + w if K < 2 else 4
                        i = 2 * I + u if I < 2 else 4
                        global_w_idx = k * n + i
                        for c in range(r):
                            W[global_w_idx][current_col + c] = W_loc[w * sI + u][c]

                current_col += r

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
