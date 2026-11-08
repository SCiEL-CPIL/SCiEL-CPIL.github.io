"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    def block_idx(b, i, j):
        row = (b // 2) * 2 + i
        col_ = (b % 2) * 2 + j
        return row * 4 + col_
    
    products = [
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ]),
        ([ (2,1), (3,1) ], [ (0,1) ]),
        ([ (0,1) ], [ (1,1), (3,-1) ]),
        ([ (3,1) ], [ (2,1), (0,-1) ]),
        ([ (0,1), (1,1) ], [ (3,1) ]),
        ([ (2,1), (0,-1) ], [ (0,1), (1,1) ]),
        ([ (1,1), (3,-1) ], [ (2,1), (3,1) ]),
    ]
    
    m_to_c = {
        0: [(0, 1), (3, 1)],
        1: [(2, 1), (3, -1)],
        2: [(1, 1), (3, 1)],
        3: [(0, 1), (2, 1)],
        4: [(0, -1), (1, 1)],
        5: [(3, 1)],
        6: [(0, 1)],
    }
    
    def build_strassen_4x4x4(base_col):
        U_entries = []
        V_entries = []
        W_entries = []
        col = base_col
        
        for m_idx in range(7):
            a_blocks = products[m_idx][0]
            b_blocks = products[m_idx][1]
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u_indices = {}
                        v_indices = {}
                        
                        for a_b, a_s in a_blocks:
                            a_idx = block_idx(a_b, i, j)
                            u_indices[a_idx] = u_indices.get(a_idx, 0) + a_s
                        
                        for b_b, b_s in b_blocks:
                            b_idx = block_idx(b_b, j, k)
                            v_indices[b_idx] = v_indices.get(b_idx, 0) + b_s
                        
                        for c_b, c_s in m_to_c[m_idx]:
                            c_idx = block_idx(c_b, k, i)
                            
                            for a_idx, a_val in u_indices.items():
                                if a_val != 0:
                                    U_entries.append((a_idx, col, a_val))
                            
                            for b_idx, b_val in v_indices.items():
                                if b_val != 0:
                                    V_entries.append((b_idx, col, b_val))
                            
                            W_entries.append((c_idx, col, c_s))
                        
                        col += 1
        
        return U_entries, V_entries, W_entries, col - base_col
    
    U1, V1, W1, r1 = build_strassen_4x4x4(0)
    U2, V2, W2, r2 = build_strassen_4x4x4(r1)
    
    total_rank = r1 + r2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for idx, col, val in U1:
        U[idx][col] += val
    for idx, col, val in V1:
        V[idx][col] += val
    for idx, col, val in W1:
        W[idx][col] += val
    
    for idx, col, val in U2:
        U[idx][col] += val
    for idx, col, val in V2:
        V[idx + 16][col] += val
    for idx, col, val in W2:
        W[idx + 16][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
