"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 decomposition
    def add_strassen(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i, col_offset):
        # Local 2x2 indices mapping to global
        # U rows: (i+j) -> u
        # V rows: (j*2+k) -> v
        # W rows: (k*2+i) -> w
        
        u_idxs = [u_offset_i * m + u_offset_j, 
                  u_offset_i * m + u_offset_j + 1,
                  (u_offset_i + 1) * m + u_offset_j, 
                  (u_offset_i + 1) * m + u_offset_j + 1]
        
        v_idxs = [v_offset_j * p + v_offset_k, 
                  v_offset_j * p + v_offset_k + 1,
                  (v_offset_j + 1) * p + v_offset_k, 
                  (v_offset_j + 1) * p + v_offset_k + 1]
        
        w_idxs = [w_offset_k * n + w_offset_i, 
                  w_offset_k * n + w_offset_i + 1,
                  (w_offset_k + 1) * n + w_offset_i, 
                  (w_offset_k + 1) * n + w_offset_i + 1]

        # Strassen U, V, W matrices (4x7, 4x7, 4x7)
        # Rows 0-3 correspond to A00, A01, A10, A11
        U_S = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_S = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        # W_S rows correspond to C00, C10, C01, C11 in order k*n+i?
        # C00: k=0, i=0 -> 0
        # C10: k=0, i=1 -> 1
        # C01: k=1, i=0 -> 2
        # C11: k=1, i=1 -> 3
        # Formulas:
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        W_S = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]

        for r in range(4):
            for c in range(7):
                val_u = U_S[r][c]
                if val_u != 0:
                    U[u_idxs[r]][col_offset + c] += val_u
                val_v = V_S[r][c]
                if val_v != 0:
                    V[v_idxs[r]][col_offset + c] += val_v
                val_w = W_S[r][c]
                if val_w != 0:
                    W[w_idxs[r]][col_offset + c] += val_w

    # Block 1: A(0:2, 0:2) x B(0:2, 0:2)
    # i=0, j=0 ; j=0, k=0 ; k=0, i=0
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A(0:2, 0:2) x B(0:2, 2:4)
    # i=0, j=0 ; j=0, k=2 ; k=2, i=0
    add_strassen(0, 0, 0, 2, 2, 0, 7)

    # Block 3: A(0:2, 0:2) x B(0:2, 4:5) -> 2x2x1
    # Schoolbook 4 terms
    # A(0,0) B(0,4) -> C(0,4)
    U[0][14] += 1
    V[4][14] += 1
    W[8][14] += 1  # k=4, i=0 -> 4*2+0 = 8
    
    # A(1,0) B(0,4) -> C(1,4)
    U[3][15] += 1
    V[4][15] += 1
    W[9][15] += 1  # k=4, i=1 -> 4*2+1 = 9
    
    # A(0,1) B(1,4) -> C(0,4)
    U[1][16] += 1
    V[9][16] += 1
    W[8][16] += 1
    
    # A(1,1) B(1,4) -> C(1,4)
    U[4][17] += 1
    V[9][17] += 1
    W[9][17] += 1

    # Block 4: A(0:2, 2:3) x B(2:3, 0:5) -> 2x1x5
    # 10 terms: A(i,2) * B(2,k) -> C(i,k)
    for i in range(2):
        for k in range(5):
            col = 18 + i * 5 + k
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
