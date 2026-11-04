import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block structure: 3x3 blocks of 2x2
    block_size = 2
    num_blocks = n // block_size # 3
    
    # Rank is num_blocks^3 * 7
    rank = (num_blocks ** 3) * 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2
    # U_coeff[step][alpha, beta]
    # V_coeff[step][beta, gamma]
    # W_coeff[step][gamma, alpha]
    
    # Step 1: (x00+x11)(y00+y11)
    # Step 2: (x10+x11)y00
    # Step 3: x00(y10-y11)
    # Step 4: x11(y01-y00)
    # Step 5: (x00+x01)y11
    # Step 6: (x10-x00)(y00+y10)
    # Step 7: (x01-x11)(y11+y01)
    
    U_strassen = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [-1, 0]]  # M7
    ]
    
    V_strassen = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 0], [1, -1]], # M3
        [[-1, 1], [0, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [1, 0]]  # M7
    ]
    
    # W coefficients: W[gamma, alpha]
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    W_strassen = [
        [[1, 0], [0, 1]], # M1: Z00=1, Z11=1
        [[0, -1], [0, 1]], # M2: Z10=1 (W[0,1]), Z11=-1 (W[1,1])
        [[0, 1], [0, 1]], # M3: Z01=1 (W[1,0]), Z11=1 (W[1,1])
        [[1, 1], [0, 0]], # M4: Z00=1 (W[0,0]), Z10=1 (W[0,1])
        [[-1, 0], [1, 0]], # M5: Z00=-1 (W[0,0]), Z01=1 (W[1,0])
        [[0, 0], [0, 1]], # M6: Z11=1 (W[1,1])
        [[1, 0], [0, 0]]  # M7: Z00=1 (W[0,0])
    ]
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for s in range(7):
                    # Fill U
                    for alpha in range(block_size):
                        for beta in range(block_size):
                            i = I * block_size + alpha
                            j = J * block_size + beta
                            U[i * m + j, col] = U_strassen[s][alpha][beta]
                            
                    # Fill V
                    for beta in range(block_size):
                        for gamma in range(block_size):
                            j = J * block_size + beta
                            k = K * block_size + gamma
                            V[j * p + k, col] = V_strassen[s][beta][gamma]
                            
                    # Fill W
                    for gamma in range(block_size):
                        for alpha in range(block_size):
                            k = K * block_size + gamma
                            i = I * block_size + alpha
                            W[k * n + i, col] = W_strassen[s][gamma][alpha]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
