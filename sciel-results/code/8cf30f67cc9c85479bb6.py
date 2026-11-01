import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen-like decomposition for 2x2x2 blocks
    # 8 blocks of 2x2x2
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(num_blocks):
        k1 = 2 * l
        k2 = 2 * l + 1
        
        # M1: (A00 + A11) * (B0_k1 + B1_k2) -> C0_k1 (+1), C1_k2 (+1)
        U[0][col] = 1
        U[3][col] = 1
        V[k1][col] = 1
        V[16 + k2][col] = 1
        W[k1 * 2 + 0][col] = 1
        W[k2 * 2 + 1][col] = 1
        col += 1
        
        # M2: (A10 + A11) * B0_k2 -> C1_k1 (+1), C1_k2 (-1)
        U[2][col] = 1
        U[3][col] = 1
        V[k2][col] = 1
        W[k1 * 2 + 1][col] = 1
        W[k2 * 2 + 1][col] = -1
        col += 1
        
        # M3: A00 * (B1_k1 - B1_k2) -> C0_k2 (+1), C1_k2 (+1)
        U[0][col] = 1
        V[16 + k1][col] = 1
        V[16 + k2][col] = -1
        W[k2 * 2 + 0][col] = 1
        W[k2 * 2 + 1][col] = 1
        col += 1
        
        # M4: A11 * (B0_k1 + B0_k2) -> C0_k1 (+1), C1_k1 (+1)
        U[3][col] = 1
        V[k1][col] = 1
        V[k2][col] = 1
        W[k1 * 2 + 0][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M5: (A00 + A01) * B1_k2 -> C0_k1 (-1), C0_k2 (+1)
        U[0][col] = 1
        U[1][col] = 1
        V[16 + k2][col] = 1
        W[k1 * 2 + 0][col] = -1
        W[k2 * 2 + 0][col] = 1
        col += 1
        
        # M6: (A10 - A00) * (B0_k1 + B0_k2) -> C1_k2 (+1)
        U[2][col] = 1
        U[0][col] = -1
        V[k1][col] = 1
        V[k2][col] = 1
        W[k2 * 2 + 1][col] = 1
        col += 1
        
        # M7: (A01 - A11) * (B1_k1 - B1_k2) -> C0_k1 (-1)
        U[1][col] = 1
        U[3][col] = -1
        V[16 + k1][col] = 1
        V[16 + k2][col] = -1
        W[k1 * 2 + 0][col] = -1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
