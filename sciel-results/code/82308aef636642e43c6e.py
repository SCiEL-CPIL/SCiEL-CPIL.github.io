import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + 4 for 4x4x1 (Schoolbook) = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition factors
    # U: (2x2) x 7, V: (2x2) x 7, W: (2x2) x 7
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, -1, -1, 1, 0, 1]
    ]
    
    # Helper to map block b and local index l to global index for 4x4
    def get_global_index(b, l, is_w=False):
        # b: 0..3 (block index)
        # l: 0..3 (local index in 2x2)
        row = l // 2
        col = l % 2
        # Block b maps to:
        # b=0: r 0-1, c 0-1
        # b=1: r 0-1, c 2-3
        # b=2: r 2-3, c 0-1
        # b=3: r 2-3, c 2-3
        block_row = b // 2
        block_col = b % 2
        g_row = 2 * block_row + row
        g_col = 2 * block_col + col
        
        if is_w:
            # W index is k*n + i. Here n=4.
            # g_row is i, g_col is k?
            # In C, indices are (i, k).
            # Local index l in W2 corresponds to (i_local, k_local).
            # So g_row is i, g_col is k.
            return g_col * n + g_row
        else:
            # U and V indices are i*m + j or j*p + k.
            # For U (A): i*m + j. g_row is i, g_col is j.
            # For V (B): j*p + k. g_row is j, g_col is k.
            # The mapping is the same structure.
            return g_row * 4 + g_col # Assuming m=4, p=4 for the 4x4 part
    
    col = 0
    for t in range(7):
        for s in range(7):
            # U part
            for b in range(4):
                if U2[b][t] != 0:
                    val_t = U2[b][t]
                    for l in range(4):
                        if U2[l][s] != 0:
                            val_s = U2[l][s]
                            g_idx = get_global_index(b, l, is_w=False)
                            U[g_idx][col] += val_t * val_s
            # V part
            for b in range(4):
                if V2[b][t] != 0:
                    val_t = V2[b][t]
                    for l in range(4):
                        if V2[l][s] != 0:
                            val_s = V2[l][s]
                            g_idx = get_global_index(b, l, is_w=False)
                            V[g_idx][col] += val_t * val_s
            # W part
            for b in range(4):
                if W2[b][t] != 0:
                    val_t = W2[b][t]
                    for l in range(4):
                        if W2[l][s] != 0:
                            val_s = W2[l][s]
                            g_idx = get_global_index(b, l, is_w=True)
                            W[g_idx][col] += val_t * val_s
            col += 1
            
    # Add 4x4x1 part
    # C[:, 4] = A * B[:, 4]
    # B[:, 4] is the 5th column. Index in V: j*5 + 4.
    # C[:, 4] indices in W: 4*4 + i = 16 + i.
    for j in range(4):
        # U: 1 at i*4 + j for all i
        for i in range(4):
            U[i * 4 + j][col] = 1
        # V: 1 at j*5 + 4
        V[j * 5 + 4][col] = 1
        # W: 1 at 16 + i for all i
        for i in range(4):
            W[16 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
