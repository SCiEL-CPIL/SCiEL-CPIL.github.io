# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    block_size = 3
    num_blocks = p // block_size
    
    rank = num_blocks * n * m * block_size
    U = [[0 for _ in range(rank)] for _ in range(n*m)]
    V = [[0 for _ in range(rank)] for _ in range(m*p)]
    W = [[0 for _ in range(rank)] for _ in range(p*n)]
    
    col = 0
    for b in range(num_blocks):
        k_off = b * block_size
        for i in range(n):
            for j in range(m):
                for k in range(block_size):
                    kk = k_off + k
                    U[i*m+j][col] = 1
                    V[j*p+kk][col] = 1
                    W[kk*n+i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
