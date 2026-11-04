"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition with 3x3 blocks.
    # The 6x6 matrix is viewed as a 2x2 matrix of 3x3 blocks.
    # We use Strassen's algorithm for the 2x2 super-matrix (Rank 7).
    # Each super-matrix multiplication involves 3x3 matrix multiplications (Rank 27).
    # Total Rank = 7 * 27 = 189.
    
    block_size = 3
    super_n = n // block_size # 2
    super_m = m // block_size # 2
    super_p = p // block_size # 2
    
    # Strassen coefficients for 2x2 matrix multiplication
    # u[t][i][j] is coeff of A[i][j] in M_t
    # v[t][j][k] is coeff of B[j][k] in M_t
    # w[t][i][k] is coeff of M_t in C[i][k]
    
    u = {}
    v = {}
    w = {}
    
    # t=1
    u[1] = {(0,0): 1, (1,1): 1}
    v[1] = {(0,0): 1, (1,1): 1}
    w[1] = {(0,0): 1, (1,1): 1}
    
    # t=2
    u[2] = {(1,0): 1, (1,1): 1}
    v[2] = {(0,0): 1}
    w[2] = {(1,0): 1, (1,1): -1}
    
    # t=3
    u[3] = {(0,0): 1}
    v[3] = {(0,1): 1, (1,1): -1}
    w[3] = {(0,1): 1, (1,1): 1}
    
    # t=4
    u[4] = {(1,1): 1}
    v[4] = {(1,0): 1, (0,0): -1}
    w[4] = {(0,0): 1, (1,0): 1}
    
    # t=5
    u[5] = {(0,0): 1, (0,1): 1}
    v[5] = {(1,1): 1}
    w[5] = {(0,0): -1, (0,1): 1}
    
    # t=6
    u[6] = {(1,0): 1, (0,0): -1}
    v[6] = {(0,0): 1, (0,1): 1}
    w[6] = {(1,1): 1}
    
    # t=7
    u[7] = {(0,1): 1, (1,1): -1}
    v[7] = {(1,0): 1, (1,1): 1}
    w[7] = {(0,0): 1}
    
    rank = 7 * block_size * block_size * block_size
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(1, 8):
        for il in range(block_size):
            for jl in range(block_size):
                for kl in range(block_size):
                    # U: A[3*ib + il, 3*jb + jl]
                    for ib in range(super_n):
                        for jb in range(super_m):
                            if (ib, jb) in u[t]:
                                idx_a = (ib * block_size + il) * m + (jb * block_size + jl)
                                U[idx_a][col] = u[t][(ib, jb)]
                    
                    # V: B[3*jb2 + jl, 3*kb + kl]
                    # Note: The jl in V must match the jl in the current column
                    for jb2 in range(super_m):
                        for kb in range(super_p):
                            if (jb2, kb) in v[t]:
                                idx_b = (jb2 * block_size + jl) * p + (kb * block_size + kl)
                                V[idx_b][col] = v[t][(jb2, kb)]
                    
                    # W: C[3*kb + kl, 3*ib + il] -> Index is k*n + i
                    # Note: The il, kl in W must match the il, kl in the current column
                    for ib in range(super_n):
                        for kb in range(super_p):
                            if (ib, kb) in w[t]:
                                idx_c = (kb * block_size + kl) * n + (ib * block_size + il)
                                W[idx_c][col] = w[t][(ib, kb)]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
