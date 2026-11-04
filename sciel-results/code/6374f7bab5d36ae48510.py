"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block sizes
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Total rank calculation
    # 8 blocks of (2,2,2) rank 7 -> 56
    # 4 blocks of (2,2,1) rank 4 -> 16
    # 4 blocks of (2,1,2) rank 1 -> 4
    # 4 blocks of (1,2,2) rank 4 -> 16
    # 2 blocks of (2,1,1) rank 2 -> 4
    # 2 blocks of (1,2,1) rank 2 -> 4
    # 2 blocks of (1,1,2) rank 2 -> 4
    # 1 block of (1,1,1) rank 1 -> 1
    # Total: 105
    
    total_rank = 105
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add factors
    def add_factors(local_U, local_V, local_W, i, j, k):
        nonlocal col_offset
        rank = len(local_U[0])
        
        # local_U is n_i x m_j
        # local_V is m_j x p_k
        # local_W is p_k x n_i
        
        n_i = sizes[i]
        m_j = sizes[j]
        p_k = sizes[k]
        
        off_i = offsets[i]
        off_j = offsets[j]
        off_k = offsets[k]
        
        for mu in range(rank):
            # U
            for ri in range(n_i):
                for cj in range(m_j):
                    val = local_U[ri][cj][mu]
                    if val != 0:
                        global_row = (off_i + ri) * m + (off_j + cj)
                        U[global_row][col_offset + mu] += val
                        
            # V
            for rj in range(m_j):
                for ck in range(p_k):
                    val = local_V[rj][ck][mu]
                    if val != 0:
                        global_row = (off_j + rj) * p + (off_k + ck)
                        V[global_row][col_offset + mu] += val
                        
            # W
            # W index is (k, i) -> col k, row i
            for rk in range(p_k):
                for ci in range(n_i):
                    val = local_W[rk][ci][mu]
                    if val != 0:
                        global_row = (off_k + rk) * n + (off_i + ci)
                        W[global_row][col_offset + mu] += val
                        
        col_offset += rank

    # Define local factors for each block type
    # Type (ni, mj, pk)
    
    def get_factors(ni, mj, pk):
        # Returns U, V, W as lists of lists of lists (rank last)
        # Or just lists of rank-length vectors for each index
        
        if ni == 2 and mj == 2 and pk == 2:
            # Strassen 2x2x2
            # U: 4x7, V: 4x7, W: 4x7
            # Indices: U(ri, cj), V(rj, ck), W(rk, ci)
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # Strassen factors
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 - M4 + M6 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M5 + M7
            
            # Map to U, V, W
            # U[ri, cj] corresponds to A[ri, cj]
            # V[rj, ck] corresponds to B[rj, ck]
            # W[rk, ci] corresponds to C[ri, rk] (note W index is k, i)
            
            # M1: A00+A11, B00+B11
            # U: (0,0)=1, (1,1)=1
            # V: (0,0)=1, (1,1)=1
            # W: (0,0)=1 (C00), (1,1)=1 (C11)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][3] = 1
            
            # M2: A10+A11, B00
            # U: (1,0)=1, (1,1)=1
            # V: (0,0)=1
            # W: (1,1)=1 (C11? No, C10? M2 contributes to C10)
            # C10 = M2 + M4. So W[1,1] += 1? No, W index is (k,i).
            # C10 is row 1, col 0. So k=0, i=1. W[0,1].
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[0][2] = 1 # C10 is (k=0, i=1) -> index 1*5+0? No, local index.
            # Local W index: (k, i). k in {0,1}, i in {0,1}.
            # C10 -> k=0, i=1. W[0,1].
            W[0][1] = 1
            
            # M3: A00, B10-B11
            # U: (0,0)=1
            # V: (1,0)=1, (1,1)=-1
            # W: C01 (k=1, i=0). W[1,0].
            U[0][2] = 1
            V[2][2] = 1; V[3][2] = -1
            W[1][0] = 1
            
            # M4: A11, B01-B00
            # U: (1,1)=1
            # V: (0,1)=1, (0,0)=-1
            # W: C00 (k=0, i=0). W[0,0]. C11 (k=1, i=1). W[1,1].
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1
            W[0][0] -= 1
            W[1][1] = 1
            
            # M5: A00+A01, B11
            # U: (0,0)=1, (0,1)=1
            # V: (1,1)=1
            # W: C01 (k=1, i=0). W[1,0]. C11 (k=1, i=1). W[1,1].
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[1][0] += 1
            W[1][1] += 1
            
            # M6: A10-A00, B00+B10
            # U: (1,0)=1, (0,0)=-1
            # V: (0,0)=1, (1,0)=1
            # W: C00 (k=0, i=0). W[0,0].
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[2][5] = 1
            W[0][0] += 1
            
            # M7: A01-A11, B01+B11
            # U: (0,1)=1, (1,1)=-1
            # V: (0,1)=1, (1,1)=1
            # W: C00 (k=0, i=0). W[0,0]. C11 (k=1, i=1). W[1,1].
            U[1][6] = 1; U[3][6] = -1
            V[1][6] = 1; V[3][6] = 1
            W[0][0] -= 1
            W[1][1] += 1
            
            return U, V, W

        elif ni == 2 and mj == 2 and pk == 1:
            # Rank 4
            U = [[0]*4 for _ in range(4)]
            V = [[0]*4 for _ in range(2)]
            W = [[0]*4 for _ in range(2)]
            
            # C0 = A00 B0 + A01 B1
            # C1 = A10 B0 + A11 B1
            
            # M1: A00 B0 -> C0
            U[0][0] = 1
            V[0][0] = 1
            W[0][0] = 1 # C0 is k=0, i=0. W[0,0].
            
            # M2: A01 B1 -> C0
            U[1][1] = 1
            V[1][1] = 1
            W[0][0] += 1
            
            # M3: A10 B0 -> C1
            U[2][2] = 1
            V[0][2] = 1
            W[0][1] = 1 # C1 is k=0, i=1. W[0,1].
            
            # M4: A11 B1 -> C1
            U[3][3] = 1
            V[1][3] = 1
            W[0][1] += 1
            
            return U, V, W

        elif ni == 2 and mj == 1 and pk == 2:
            # Rank 1
            U = [[0]*1 for _ in range(2)]
            V = [[0]*1 for _ in range(2)]
            W = [[0]*1 for _ in range(4)]
            
            U[0][0] = 1
            U[1][0] = 1
            V[0][0] = 1
            V[0][1] = 1
            W[0][0] = 1 # C00 (k=0, i=0)
            W[1][0] = 1 # C01 (k=1, i=0)
            W[0][1] = 1 # C10 (k=0, i=1)
            W[1][1] = 1 # C11 (k=1, i=1)
            
            return U, V, W

        elif ni == 1 and mj == 2 and pk == 2:
            # Rank 4
            U = [[0]*4 for _ in range(2)]
            V = [[0]*4 for _ in range(4)]
            W = [[0]*4 for _ in range(2)]
            
            # C0 = A0 B00 + A1 B10
            # C1 = A0 B01 + A1 B11
            
            # M1: A0 B00 -> C0
            U[0][0] = 1
            V[0][0] = 1
            W[0][0] = 1
            
            # M2: A1 B10 -> C0
            U[1][1] = 1
            V[2][1] = 1
            W[0][0] += 1
            
            # M3: A0 B01 -> C1
            U[0][2] = 1
            V[1][2] = 1
            W[1][0] = 1
            
            # M4: A1 B11 -> C1
            U[1][3] = 1
            V[3][3] = 1
            W[1][0] += 1
            
            return U, V, W

        elif ni == 2 and mj == 1 and pk == 1:
            # Rank 2
            U = [[0]*2 for _ in range(2)]
            V = [[0]*2 for _ in range(1)]
            W = [[0]*2 for _ in range(2)]
            
            # C0 = A0 B
            # C1 = A1 B
            
            U[0][0] = 1
            V[0][0] = 1
            W[0][0] = 1
            
            U[1][1] = 1
            V[0][1] = 1
            W[0][1] = 1
            
            return U, V, W

        elif ni == 1 and mj == 2 and pk == 1:
            # Rank 2
            U = [[0]*2 for _ in range(2)]
            V = [[0]*2 for _ in range(2)]
            W = [[0]*2 for _ in range(1)]
            
            # C = A0 B0 + A1 B1
            
            U[0][0] = 1
            V[0][0] = 1
            W[0][0] = 1
            
            U[1][1] = 1
            V[1][1] = 1
            W[0][0] += 1
            
            return U, V, W

        elif ni == 1 and mj == 1 and pk == 2:
            # Rank 2
            U = [[0]*2 for _ in range(1)]
            V = [[0]*2 for _ in range(2)]
            W = [[0]*2 for _ in range(2)]
            
            # C0 = A B0
            # C1 = A B1
            
            U[0][0] = 1
            V[0][0] = 1
            W[0][0] = 1
            
            U[0][1] = 1
            V[1][1] = 1
            W[1][0] = 1
            
            return U, V, W

        elif ni == 1 and mj == 1 and pk == 1:
            # Rank 1
            U = [[0]*1 for _ in range(1)]
            V = [[0]*1 for _ in range(1)]
            W = [[0]*1 for _ in range(1)]
            
            U[0][0] = 1
            V[0][0] = 1
            W[0][0] = 1
            
            return U, V, W

        else:
            raise ValueError(f"Unknown block type {ni}x{mj}x{pk}")

    # Iterate over blocks
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = sizes[i]
                mj = sizes[j]
                pk = sizes[k]
                
                local_U, local_V, local_W = get_factors(ni, mj, pk)
                add_factors(local_U, local_V, local_W, i, j, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
