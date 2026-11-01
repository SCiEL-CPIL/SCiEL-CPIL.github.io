"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 4
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    for b in range(num_blocks):
        base = b * 7
        # W indices for C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
        w_00 = 2 * b * n + 0  # C[0,2b]
        w_01 = (2 * b + 1) * n + 0  # C[0,2b+1]
        w_10 = 2 * b * n + 1  # C[1,2b]
        w_11 = (2 * b + 1) * n + 1  # C[1,2b+1]
        
        # B indices
        b_e = 0 * p + 2 * b      # B[0,2b]
        b_f = 0 * p + 2 * b + 1  # B[0,2b+1]
        b_g = 1 * p + 2 * b      # B[1,2b]
        b_h = 1 * p + 2 * b + 1  # B[1,2b+1]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1
        U[3][base + 0] += 1
        V[b_e][base + 0] += 1
        V[b_h][base + 0] += 1
        W[w_00][base + 0] += 1
        W[w_11][base + 0] += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1
        U[3][base + 1] += 1
        V[b_e][base + 1] += 1
        W[w_00][base + 1] += 1
        W[w_11][base + 1] -= 1
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1
        V[b_f][base + 2] += 1
        V[b_h][base + 2] -= 1
        W[w_01][base + 2] += 1
        W[w_11][base + 2] += 1
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1
        V[b_g][base + 3] += 1
        V[b_e][base + 3] -= 1
        W[w_00][base + 3] += 1
        W[w_11][base + 3] += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1
        U[1][base + 4] += 1
        V[b_h][base + 4] += 1
        W[w_00][base + 4] -= 1
        W[w_10][base + 4] += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1
        U[0][base + 5] -= 1
        V[b_e][base + 5] += 1
        V[b_f][base + 5] += 1
        W[w_11][base + 5] += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1
        U[3][base + 6] -= 1
        V[b_g][base + 6] += 1
        V[b_h][base + 6] += 1
        W[w_00][base + 6] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
