import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen patterns for 2x2 multiplication
    # Each pattern is a list of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (i, j, val)
    # V_coeffs: list of (j, k_local, val)
    # W_coeffs: list of (k_local, i, val)
    
    patterns = [
        # P1
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # P2
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, -1)]
        },
        # P3
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(1, 0, 1), (1, 1, 1)]
        },
        # P4
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]
        },
        # P5
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)]
        },
        # P6
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # P7
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    for q in range(num_blocks):
        for s in range(rank_per_block):
            col = q * rank_per_block + s
            pat = patterns[s]
            
            for i, j, val in pat['U']:
                U[i * m + j][col] += val
                
            for j, k_local, val in pat['V']:
                k_global = 2 * q + k_local
                V[j * p + k_global][col] += val
                
            for k_local, i, val in pat['W']:
                k_global = 2 * q + k_local
                W[k_global * n + i][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
