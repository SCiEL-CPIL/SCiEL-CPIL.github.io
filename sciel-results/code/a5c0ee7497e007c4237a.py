import time
import sys

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into p//2 blocks of 2x2x2 multiplications
    # Each 2x2x2 multiplication uses Strassen's algorithm with rank 7
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Local factors for 2x2x2 Strassen decomposition
    # Indices: i,j in {0,1} for A; j,r in {0,1} for B; r,i in {0,1} for C
    # U_local[i*2+j, s]
    # V_local[j*2+r, s]
    # W_local[r*2+i, s]
    
    U_local = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_local = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W_local = [
        [1, 0, 1, 1, -1, 0, 1], # C00 (r=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (r=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (r=1, i=0)
        [1, -1, 0, 0, 0, 1, 0]  # C11 (r=1, i=1)
    ]
    
    for g in range(num_blocks):
        col_offset = g * rank_per_block
        
        # Fill U: same for all blocks
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for s in range(rank_per_block):
                    U[u_idx][col_offset + s] = U_local[u_idx][s]
        
        # Fill V: sparse, only for k in {2g, 2g+1}
        for j in range(m):
            for r in range(2):
                k = 2 * g + r
                v_idx = j * p + k
                v_local_idx = j * 2 + r
                for s in range(rank_per_block):
                    V[v_idx][col_offset + s] = V_local[v_local_idx][s]
        
        # Fill W: sparse, only for k in {2g, 2g+1}
        for r in range(2):
            k = 2 * g + r
            for i in range(n):
                w_idx = k * n + i
                w_local_idx = r * 2 + i
                for s in range(rank_per_block):
                    W[w_idx][col_offset + s] = W_local[w_local_idx][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
