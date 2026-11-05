"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: 6x6 = 2x2 blocks of 3x3
    # Use Strassen (rank 7) for 2x2 block level
    # Use schoolbook (rank 27) for 3x3 inner multiplications
    # Total rank = 7 * 27 = 189
    
    block_size = 3
    num_blocks = 2
    
    # Strassen's 7 products for 2x2 matrix multiplication
    strassen_products = [
        ([[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[0,0],[0,1]], [[0,-1],[1,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,-1],[1,0]], [[1,1],[0,0]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]]),
    ]
    
    C_coeff = [
        [[1,0],[0,1]],
        [[0,0],[1,-1]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]],
    ]
    
    rank = 7 * 27  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(7):
        A_c = strassen_products[s][0]
        B_c = strassen_products[s][1]
        C_c = C_coeff[s]
        for i_prime in range(block_size):
            for j_prime in range(block_size):
                for k_prime in range(block_size):
                    for a_blk in range(num_blocks):
                        for b_blk in range(num_blocks):
                            val = A_c[a_blk][b_blk]
                            if val != 0:
                                i_idx = a_blk * block_size + i_prime
                                j_idx = b_blk * block_size + j_prime
                                U[i_idx * m + j_idx][col] = val
                    for c_blk in range(num_blocks):
                        for d_blk in range(num_blocks):
                            val = B_c[c_blk][d_blk]
                            if val != 0:
                                j_idx = c_blk * block_size + j_prime
                                k_idx = d_blk * block_size + k_prime
                                V[j_idx * p + k_idx][col] = val
                    for i_blk in range(num_blocks):
                        for k_blk in range(num_blocks):
                            val = C_c[i_blk][k_blk]
                            if val != 0:
                                i_idx = i_blk * block_size + i_prime
                                k_idx = k_blk * block_size + k_prime
                                W[k_idx * n + i_idx][col] = val
                    col += 1
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        pass  # Should not happen, but safety
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
