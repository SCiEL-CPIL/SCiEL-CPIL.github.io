"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This gives rank 48 for 2x4 * 4x6.
    # We attempt to find a lower-rank decomposition via Strassen-like block techniques.
    # For 2x4 * 4x6, we can split B into 2x2 blocks in the row dimension and 2x2 in col dimension.
    # A is 2x4, B is 4x6, C is 2x6.
    # Let A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [B_TL, B_TM, B_TR; B_BL, B_BM, B_BR] where each block is 2x2.
    # C = A_L * B_T + A_R * B_B where B_T = [B_TL, B_TM, B_TR], B_B = [B_BL, B_BM, B_BR].
    # C[:,0:2] = A_L * B_TL + A_R * B_BL
    # C[:,2:4] = A_L * B_TM + A_R * B_BM
    # C[:,4:6] = A_L * B_TR + A_R * B_BR
    # Each 2x2*2x2 + 2x2*2x2 can be done with Strassen for each pair, but they share A_L and A_R.
    # Actually, C[:,0:2] = A_L*B_TL + A_R*B_BL. This is a sum of two matrix products.
    # We can use the fact that (A_L+A_R)*(B_TL+B_BL) = A_L*B_TL + A_L*B_BL + A_R*B_TL + A_R*B_BL
    # But we need A_L*B_TL + A_R*B_BL. This doesn't directly help.
    
    # Let's just return the schoolbook decomposition which is guaranteed correct.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
