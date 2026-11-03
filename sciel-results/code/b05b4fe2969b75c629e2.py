import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 8
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    def add_strassen_block(rank_offset):
        i0, j0, k0 = 0, 0, 0
        i1, j1, k1 = 1, 1, 1
        
        u_i0_j0 = i0 * m + j0
        u_i0_j1 = i0 * m + j1
        u_i1_j0 = i1 * m + j0
        u_i1_j1 = i1 * m + j1
        
        v_j0_k0 = j0 * p + k0
        v_j0_k1 = j0 * p + k1
        v_j1_k0 = j1 * p + k0
        v_j1_k1 = j1 * p + k1
        
        w_k0_i0 = k0 * n + i0
        w_k0_i1 = k0 * n + i1
        w_k1_i0 = k1 * n + i0
        w_k1_i1 = k1 * n + i1
        
        # M1
        U[u_i0_j0][rank_offset] += 1
        U[u_i1_j1][rank_offset] += 1
        V[v_j0_k1][rank_offset] += 1
        V[v_j1_k1][rank_offset] += 1
        W[w_k0_i0][rank_offset] += 1
        W[w_k1_i1][rank_offset] += 1
        
        # M2
        U[u_i1_j0][rank_offset + 1] += 1
        U[u_i1_j1][rank_offset + 1] += 1
        V[v_j0_k0][rank_offset + 1] += 1
        W[w_k0_i1][rank_offset + 1] += 1
        
        # M3
        U[u_i0_j0][rank_offset + 2] += 1
        V[v_j0_k0][rank_offset + 2] += 1
        V[v_j1_k0][rank_offset + 2] += -1
        W[w_k1_i0][rank_offset + 2] += 1
        W[w_k1_i1][rank_offset + 2] += 1
        
        # M4
        U[u_i1_j1][rank_offset + 3] += 1
        V[v_j1_k1][rank_offset + 3] += 1
        V[v_j0_k1][rank_offset + 3] += -1
        W[w_k0_i0][rank_offset + 3] += 1
        W[w_k0_i1][rank_offset + 3] += 1
        
        # M5
        U[u_i0_j0][rank_offset + 4] += 1
        U[u_i0_j1][rank_offset + 4] += 1
        V[v_j1_k1][rank_offset + 4] += 1
        W[w_k1_i0][rank_offset + 4] += 1
        W[w_k0_i0][rank_offset + 4] += -1
        
        # M6
        U[u_i1_j0][rank_offset + 5] += 1
        U[u_i0_j0][rank_offset + 5] += -1
        V[v_j0_k0][rank_offset + 5] += 1
        V[v_j0_k1][rank_offset + 5] += 1
        W[w_k1_i1][rank_offset + 5] += 1
        
        # M7
        U[u_i0_j1][rank_offset + 6] += 1
        U[u_i1_j1][rank_offset + 6] += -1
        V[v_j1_k0][rank_offset + 6] += 1
        V[v_j1_k1][rank_offset + 6] += 1
        W[w_k0_i0][rank_offset + 6] += 1
        W[w_k1_i1][rank_offset + 6] += 1

    # Block 1: 4 Strassen blocks for k=0..7
    for g in range(4):
        k_base = g * 2
        # We need to call add_strassen_block but it's hardcoded for k0=0.
        # Let's make it flexible or just call it with offsets.
        # Actually, the indices k0, k1 depend on g.
        # I will refactor add_strassen_block to take k0.
        
    # Refactored approach:
    def add_strassen_block_flex(i0, j0, k0, rank_offset):
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        
        u_i0_j0 = i0 * m + j0
        u_i0_j1 = i0 * m + j1
        u_i1_j0 = i1 * m + j0
        u_i1_j1 = i1 * m + j1
        
        v_j0_k0 = j0 * p + k0
        v_j0_k1 = j0 * p + k1
        v_j1_k0 = j1 * p + k0
        v_j1_k1 = j1 * p + k1
        
        w_k0_i0 = k0 * n + i0
        w_k0_i1 = k0 * n + i1
        w_k1_i0 = k1 * n + i0
        w_k1_i1 = k1 * n + i1
        
        # M1
        U[u_i0_j0][rank_offset] += 1
        U[u_i1_j1][rank_offset] += 1
        V[v_j0_k1][rank_offset] += 1
        V[v_j1_k1][rank_offset] += 1
        W[w_k0_i0][rank_offset] += 1
        W[w_k1_i1][rank_offset] += 1
        
        # M2
        U[u_i1_j0][rank_offset + 1] += 1
        U[u_i1_j1][rank_offset + 1] += 1
        V[v_j0_k0][rank_offset + 1] += 1
        W[w_k0_i1][rank_offset + 1] += 1
        
        # M3
        U[u_i0_j0][rank_offset + 2] += 1
        V[v_j0_k0][rank_offset + 2] += 1
        V[v_j1_k0][rank_offset + 2] += -1
        W[w_k1_i0][rank_offset + 2] += 1
        W[w_k1_i1][rank_offset + 2] += 1
        
        # M4
        U[u_i1_j1][rank_offset + 3] += 1
        V[v_j1_k1][rank_offset + 3] += 1
        V[v_j0_k1][rank_offset + 3] += -1
        W[w_k0_i0][rank_offset + 3] += 1
        W[w_k0_i1][rank_offset + 3] += 1
        
        # M5
        U[u_i0_j0][rank_offset + 4] += 1
        U[u_i0_j1][rank_offset + 4] += 1
        V[v_j1_k1][rank_offset + 4] += 1
        W[w_k1_i0][rank_offset + 4] += 1
        W[w_k0_i0][rank_offset + 4] += -1
        
        # M6
        U[u_i1_j0][rank_offset + 5] += 1
        U[u_i0_j0][rank_offset + 5] += -1
        V[v_j0_k0][rank_offset + 5] += 1
        V[v_j0_k1][rank_offset + 5] += 1
        W[w_k1_i1][rank_offset + 5] += 1
        
        # M7
        U[u_i0_j1][rank_offset + 6] += 1
        U[u_i1_j1][rank_offset + 6] += -1
        V[v_j1_k0][rank_offset + 6] += 1
        V[v_j1_k1][rank_offset + 6] += 1
        W[w_k0_i0][rank_offset + 6] += 1
        W[w_k1_i1][rank_offset + 6] += 1

    current_rank = 0
    for g in range(4):
        k_base = g * 2
        add_strassen_block_flex(0, 0, k_base, current_rank)
        current_rank += 7
        
    # Block 2: i in {0,1}, j=2, k in 0..7
    for i in range(2):
        for k in range(8):
            U[i * m + 2][current_rank] = 1
            V[2 * p + k][current_rank] = 1
            W[k * n + i][current_rank] = 1
            current_rank += 1
            
    # Block 3: i=2, j in 0..2, k in 0..7
    for j in range(3):
        for k in range(8):
            U[2 * m + j][current_rank] = 1
            V[j * p + k][current_rank] = 1
            W[k * n + 2][current_rank] = 1
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
