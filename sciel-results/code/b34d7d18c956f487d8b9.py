"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 68
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # 2x2x8 decomposition using Strassen (2x2x2) tensor Identity (1x1x4)
    # Strassen 2x2x2 terms:
    # u, v, w are vectors of size 4 (indices i*2+j, j*2+k, k*2+i)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]),
    ]
    
    # Identity 1x1x4 terms:
    # u, v, w are vectors of size 1, 4, 4
    # u is always [1]
    # v is basis vector of size 4
    # w is basis vector of size 4
    for k_id in range(4):
        v_id = [0, 0, 0, 0]
        v_id[k_id] = 1
        w_id = [0, 0, 0, 0]
        w_id[k_id] = 1
        
        for u_s, v_s, w_s in strassen_terms:
            # Map to global indices
            # i, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = u_s[i * 2 + j]
            
            # j in 0..1, K in 0..7
            # K = k * 4 + k_id
            for j in range(2):
                for k in range(2):
                    K = k * 4 + k_id
                    V[j * p + K][col] = v_s[j * 2 + k] * v_id[k_id] # v_id is 1 only at k_id
            
            # K in 0..7, i in 0..1
            for k in range(2):
                for i in range(2):
                    K = k * 4 + k_id
                    W[K * n + i][col] = w_s[k * 2 + i] * w_id[k_id]
            
            col += 1
    
    # Block 2: A[0:2, 2] * B[2, 0:8]
    # 2x1x8. Rank 16.
    # Terms: for each i in 0..1, k in 0..7
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 3: A[2, 0:2] * B[0:2, 0:8]
    # 1x2x8. Rank 16.
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # Block 4: A[2, 2] * B[2, 0:8]
    # 1x1x8. Rank 8.
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
