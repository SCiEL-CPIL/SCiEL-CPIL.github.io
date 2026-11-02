"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition:
    # 6 blocks of 2x2 (Strassen, rank 7)
    # 1 block of 2x1 (Schoolbook, rank 4)
    # Total rank = 6 * 7 + 4 = 46
    
    num_pairs = p // 2
    num_singles = p % 2
    rank = num_pairs * 7 + num_singles * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Strassen 2x2 decomposition vectors
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Note: W index is k*n + i. Local k=0,1. i=0,1.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    strassen_U = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [1, 0, -1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # P1: B00, B11
        [1, 0, 0, 0], # P2: B00
        [0, 0, 1, -1],# P3: B10, B11
        [1, -1, 0, 0],# P4: B00, B01
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00, B01
        [0, 0, 1, 1]  # P7: B10, B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # P1: C00, C11
        [0, 0, 1, -1],# P2: C10, C11
        [0, 1, 0, 1], # P3: C01, C11
        [1, 0, 1, 0], # P4: C00, C10
        [-1, 1, 0, 0],# P5: C00, C01
        [0, 0, 0, 1], # P6: C11
        [-1, 0, 0, 0] # P7: C00
    ]
    
    # 2x1 decomposition vectors
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V local indices: 0:(0,0), 1:(1,0)  (j=0,1; k=0)
    # W local indices: 0:(0,0), 1:(0,1)  (k=0; i=0,1)
    
    single_U = [
        [1, 0, 0, 0], # Q1: A00
        [0, 1, 0, 0], # Q2: A01
        [0, 0, 1, 0], # Q3: A10
        [0, 0, 0, 1]  # Q4: A11
    ]
    
    single_V = [
        [1, 0], # Q1: B00
        [0, 1], # Q2: B10
        [1, 0], # Q3: B00
        [0, 1]  # Q4: B10
    ]
    
    single_W = [
        [1, 0], # Q1: C00
        [1, 0], # Q2: C00
        [0, 1], # Q3: C10
        [0, 1]  # Q4: C10
    ]
    
    # Fill blocks
    for b in range(num_pairs):
        k_start = 2 * b
        # 7 terms
        for t in range(7):
            col = col_offset + t
            
            # U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    U[u_idx][col] = strassen_U[t][u_idx]
            
            # V
            for j in range(m):
                for k_local in range(2):
                    k_global = k_start + k_local
                    v_idx = j * p + k_global
                    v_local = j * 2 + k_local
                    V[v_idx][col] = strassen_V[t][v_local]
            
            # W
            for k_local in range(2):
                k_global = k_start + k_local
                for i in range(n):
                    w_idx = k_global * n + i
                    w_local = k_local * n + i
                    W[w_idx][col] = strassen_W[t][w_local]
        
        col_offset += 7
    
    for s in range(num_singles):
        k_start = 2 * num_pairs + s
        # 4 terms
        for t in range(4):
            col = col_offset + t
            
            # U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    U[u_idx][col] = single_U[t][u_idx]
            
            # V
            for j in range(m):
                k_global = k_start
                v_idx = j * p + k_global
                v_local = j
                V[v_idx][col] = single_V[t][v_local]
            
            # W
            k_global = k_start
            for i in range(n):
                w_idx = k_global * n + i
                w_local = i
                W[w_idx][col] = single_W[t][w_local]
        
        col_offset += 4
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
