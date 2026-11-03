import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_start, i_map, j_map, k_map):
        # i_map: local -> global
        # j_map: local -> global
        # k_map: local -> global
        
        # M1
        c = col_start
        U[i_map[0]*m + j_map[0]][c] += 1
        U[i_map[1]*m + j_map[1]][c] += 1
        V[j_map[0]*p + k_map[0]][c] += 1
        V[j_map[1]*p + k_map[1]][c] += 1
        W[k_map[0]*n + i_map[0]][c] += 1
        W[k_map[1]*n + i_map[1]][c] += 1
        
        # M2
        c = col_start + 1
        U[i_map[1]*m + j_map[0]][c] += 1
        U[i_map[1]*m + j_map[1]][c] += 1
        V[j_map[0]*p + k_map[0]][c] += 1
        W[k_map[0]*n + i_map[1]][c] += 1
        
        # M3
        c = col_start + 2
        U[i_map[0]*m + j_map[0]][c] += 1
        V[j_map[1]*p + k_map[0]][c] += 1
        V[j_map[1]*p + k_map[1]][c] += -1
        W[k_map[1]*n + i_map[0]][c] += 1
        
        # M4
        c = col_start + 3
        U[i_map[1]*m + j_map[1]][c] += 1
        V[j_map[0]*p + k_map[1]][c] += 1
        V[j_map[0]*p + k_map[0]][c] += -1
        W[k_map[0]*n + i_map[0]][c] += 1
        W[k_map[0]*n + i_map[1]][c] += 1
        
        # M5
        c = col_start + 4
        U[i_map[0]*m + j_map[0]][c] += 1
        U[i_map[0]*m + j_map[1]][c] += 1
        V[j_map[1]*p + k_map[1]][c] += 1
        W[k_map[0]*n + i_map[0]][c] += -1
        W[k_map[1]*n + i_map[0]][c] += 1
        
        # M6
        c = col_start + 5
        U[i_map[1]*m + j_map[0]][c] += 1
        U[i_map[0]*m + j_map[0]][c] += -1
        V[j_map[0]*p + k_map[0]][c] += 1
        V[j_map[1]*p + k_map[0]][c] += 1
        W[k_map[1]*n + i_map[1]][c] += 1
        
        # M7
        c = col_start + 6
        U[i_map[1]*m + j_map[1]][c] += 1
        U[i_map[0]*m + j_map[1]][c] += -1
        V[j_map[0]*p + k_map[1]][c] += 1
        V[j_map[1]*p + k_map[1]][c] += 1
        W[k_map[0]*n + i_map[0]][c] += 1
        W[k_map[1]*n + i_map[1]][c] += 1

    # Block 1: A1 B1L
    add_strassen_block(0, {0:0, 1:1}, {0:0, 1:1}, {0:0, 1:1})
    # Block 2: A1 B1R
    add_strassen_block(7, {0:0, 1:1}, {0:0, 1:1}, {0:2, 1:3})
    # Block 3: A2 B2L
    add_strassen_block(14, {0:0, 1:1}, {0:2, 1:3}, {0:0, 1:1})
    # Block 4: A2 B2R
    add_strassen_block(21, {0:0, 1:1}, {0:2, 1:3}, {0:2, 1:3})
    
    return {"U": U, "V": V, "W": W}
