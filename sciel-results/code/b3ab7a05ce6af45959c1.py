"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Partition: 4x5x6 = (4x4)x(4x6) + (4x1)x(1x6)
    # (4x4)x(4x6) = (4x4)x(4x4) + (4x4)x(4x2)
    # All sub-problems use schoolbook except 4x1x6 which is rank 1
    
    rank_444 = 64
    rank_442 = 32
    rank_416 = 1
    total_rank = rank_444 + rank_442 + rank_416
    
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    # 4x4x4 schoolbook
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    # 4x4x2 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*5+j][rank_444 + col - 64 + i*8 + j*2 + k] = 1  # This is getting messy
    
    # Let me redo this more carefully
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    col = 0
    # 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5+j][col] = 1
                V[j*6+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    # 4x4x2
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*5+j][col] = 1
                V[j*6+4+k][col] = 1
                W[4+k*4+i][col] = 1
                col += 1
    
    # 4x1x6
    for i in range(4):
        U[i*5+4][col] = 1
    V[4*6][col] = 1
    for k in range(6):
        for i in range(4):
            W[k*4+i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
