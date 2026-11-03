"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition 2, 2, 1
    # Indices: 0,1 in block 0; 2,3 in block 1; 4 in block 2
    blocks = [(0, 2), (2, 4), (4, 5)]
    
    # Total rank calculation
    # 8 terms of 2x2x2 (rank 8) -> 64
    # 10 terms of mixed 2,1 (rank 4) -> 40
    # 6 terms of mixed 1,1 (rank 2) -> 12
    # 1 term of 1x1x1 (rank 1) -> 1
    # Total 117
    
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                start_i, end_i = blocks[bi]
                start_k, end_k = blocks[bk]
                start_j, end_j = blocks[bj]
                
                s_i = end_i - start_i
                s_k = end_k - start_k
                s_j = end_j - start_j
                
                # Schoolbook for this block
                for u in range(s_i):
                    for v in range(s_k):
                        for w in range(s_j):
                            gi = start_i + u
                            gk = start_k + v
                            gj = start_j + w
                            
                            # U index: i * m + k (row i of A, col k of A)
                            u_idx = gi * m + gk
                            # V index: k * p + j (row k of B, col j of B)
                            v_idx = gk * p + gj
                            # W index: j * n + i (row j of C, col i of C)
                            w_idx = gj * n + gi
                            
                            U[u_idx][col] = 1
                            V[v_idx][col] = 1
                            W[w_idx][col] = 1
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
