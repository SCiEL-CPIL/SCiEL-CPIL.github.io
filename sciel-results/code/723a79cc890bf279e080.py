"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Exploit p = 6 = 3 * 2 by splitting into two 2x3 by 3x3 multiplications
    # Rank = 2 * (n * m * 3) = 2 * 18 = 36? No, each 2x3 by 3x3 has rank 6, total 12
    # Actually: split B into two 3x3 blocks, each multiplied by A (2x3)
    # Block 1: A * B[:, 0:3] -> 2x3 result, rank 6
    # Block 2: A * B[:, 3:6] -> 2x3 result, rank 6
    # Total rank = 12
    
    p1 = p // 2  # 3
    rank = n * m * p1 * 2  # 2 * 3 * 3 * 2 = 36? No...
    # Each 2x3 by 3x3 multiplication has rank n*m*p1 = 2*3*3 = 6
    # Two such multiplications: total rank = 12
    rank = 2 * n * m * p1  # 2 * 6 = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in [0, p1)
    for r in range(n * m * p1):
        i = r // (m * p1)
        j = (r // p1) % m
        k = r % p1
        
        # U index: i*m + j
        U[i * m + j][r] = 1
        # V index: j*p + k
        V[j * p + k][r] = 1
        # W index: k*n + i
        W[k * n + i][r] = 1
    
    # Block 2: k in [p1, p)
    offset = n * m * p1
    for r in range(n * m * p1):
        i = r // (m * p1)
        j = (r // p1) % m
        k = r % p1 + p1  # offset by p1
        
        # U index: i*m + j (same as block 1)
        U[i * m + j][offset + r] = 1
        # V index: j*p + k
        V[j * p + k][offset + r] = 1
        # W index: k*n + i
        W[k * n + i][offset + r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
