import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    deadline = time.time() + 110
    
    def schoolbook(n_, m_, p_):
        rank_ = n_ * m_ * p_
        U_ = [[0]*rank_ for _ in range(n_*m_)]
        V_ = [[0]*rank_ for _ in range(m_*p_)]
        W_ = [[0]*rank_ for _ in range(p_*n_)]
        col = 0
        for i_ in range(n_):
            for j_ in range(m_):
                for k_ in range(p_):
                    U_[i_*m_ + j_][col] = 1
                    V_[j_*p_ + k_][col] = 1
                    W_[k_*n_ + i_][col] = 1
                    col += 1
        return U_, V_, W_, rank_
    
    def strassen_4x4x4():
        rank_ = 49
        U_ = [[0]*rank_ for _ in range(16)]
        V_ = [[0]*rank_ for _ in range(16)]
        W_ = [[0]*rank_ for _ in range(16)]
        
        A_forms_2x2 = [
            [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
            [1,1,0,0],[-1,0,1,0],[0,1,-1,0]
        ]
        B_forms_2x2 = [
            [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
            [0,0,0,1],[1,1,0,0],[0,0,1,1]
        ]
        C_forms_2x2 = [
            [1,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,1,0],
            [-1,1,0,0],[0,0,0,1],[1,0,0,-1]
        ]
        A_forms_block = [
            [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
            [1,1,0,0],[-1,0,1,0],[0,1,0,-1]
        ]
        B_forms_block = [
            [1,0,0,1],[1,0,0,0],[0,1,0,-1],[0,0,1,-1],
            [0,0,0,1],[1,1,0,0],[0,0,1,1]
        ]
        C_forms_block = [
            [1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],
            [-1,1,0,0],[0,0,0,1],[1,0,0,-1]
        ]
        block_A_indices = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
        block_B_indices = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
        block_C_indices = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
        
        col = 0
        for s in range(7):
            for r in range(7):
                for b in range(4):
                    if A_forms_block[s][b] != 0:
                        for pos, idx in enumerate(block_A_indices[b]):
                            U_[idx][col] += A_forms_block[s][b] * A_forms_2x2[r][pos]
                    if B_forms_block[s][b] != 0:
                        for pos, idx in enumerate(block_B_indices[b]):
                            V_[idx][col] += B_forms_block[s][b] * B_forms_2x2[r][pos]
                    if C_forms_block[s][b] != 0:
                        for pos, idx in enumerate(block_C_indices[b]):
                            W_[idx][col] += C_forms_block[s][b] * C_forms_2x2[r][pos]
                col += 1
        return U_, V_, W_, rank_
    
    # Build 4x4x4 part with Strassen
    if time.time() > deadline - 5:
        U1, V1, W1, rank1 = schoolbook(4, 4, 4)
    else:
        U1, V1, W1, rank1 = strassen_4x4x4()
    
    # Build 4x4x3 part with schoolbook
    U2, V2, W2, rank2 = schoolbook(4, 4, 3)
    
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Embed 4x4x4 part
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank1):
                V[j*7+k][r] = V1[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank1):
                W[k*4+i][r] = W1[k*4+i][r]
    
    # Embed 4x4x3 part
    for i in range(16):
        for r in range(rank2):
            U[i][rank1+r] = U2[i][r]
    for j in range(4):
        for k in range(3):
            for r in range(rank2):
                V[j*7+(4+k)][rank1+r] = V2[j*3+k][r]
    for k in range(3):
        for i in range(4):
            for r in range(rank2):
                W[(4+k)*4+i][rank1+r] = W2[k*4+i][r]
    
    return {"U": U, "V": V, "W": W}

