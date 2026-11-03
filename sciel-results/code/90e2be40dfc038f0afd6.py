import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get 2x2 Strassen decomposition
    def get_2x2_strassen():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: (x00+x11)(y00+y11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (x10+x11)y00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3: x00(y01-y11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4: x11(y10-y00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (x00+x01)y11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6: (x10-x00)(y00+y01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (x01-x11)(y10+y11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    U2, V2, W2 = get_2x2_strassen()
    
    # Construct 4x4x4 decomposition
    # Rank 49
    U49 = [[0]*49 for _ in range(16)]
    V49 = [[0]*49 for _ in range(16)]
    W49 = [[0]*49 for _ in range(16)]
    
    # Define Strassen terms
    # Each term is (A_map, B_map)
    # A_map is a list of 4 lists of (global_idx, coeff)
    # B_map is a list of 4 lists of (global_idx, coeff)
    
    # Helper to create map
    def create_map(blocks):
        # blocks: list of 4 tuples (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns list of 4 lists
        map_list = [[], [], [], []]
        for b_idx, coeff in blocks:
            base = b_idx * 4
            for k in range(4):
                map_list[k].append((base + k, coeff))
        return map_list

    # Term 1: (A00+A11)(B00+B11)
    A1 = create_map([(0, 1), (3, 1)])
    B1 = create_map([(0, 1), (3, 1)])
    
    # Term 2: (A10+A11)B00
    A2 = create_map([(2, 1), (3, 1)])
    B2 = create_map([(0, 1)])
    
    # Term 3: A00(B01-B11)
    A3 = create_map([(0, 1)])
    B3 = create_map([(1, 1), (3, -1)])
    
    # Term 4: A11(B10-B00)
    A4 = create_map([(3, 1)])
    B4 = create_map([(2, 1), (0, -1)])
    
    # Term 5: (A00+A01)B11
    A5 = create_map([(0, 1), (1, 1)])
    B5 = create_map([(3, 1)])
    
    # Term 6: (A10-A00)(B00+B01)
    A6 = create_map([(2, 1), (0, -1)])
    B6 = create_map([(0, 1), (1, 1)])
    
    # Term 7: (A01-A11)(B10+B11)
    A7 = create_map([(1, 1), (3, -1)])
    B7 = create_map([(2, 1), (3, 1)])
    
    terms = [
        (A1, B1),
        (A2, B2),
        (A3, B3),
        (A4, B4),
        (A5, B5),
        (A6, B6),
        (A7, B7)
    ]
    
    # Combination coefficients c[bi][bk][r]
    # bi: block row (0,1), bk: block col (0,1)
    c = [[[0]*7 for _ in range(2)] for _ in range(2)]
    
    # C00 = P1 + P4 - P5 + P7
    c[0][0][0] = 1
    c[0][0][3] = 1
    c[0][0][4] = -1
    c[0][0][6] = 1
    
    # C01 = P3 + P5
    c[0][1][2] = 1
    c[0][1][4] = 1
    
    # C10 = P2 + P4
    c[1][0][1] = 1
    c[1][0][3] = 1
    
    # C11 = P1 - P2 + P3 + P6
    c[1][1][0] = 1
    c[1][1][1] = -1
    c[1][1][2] = 1
    c[1][1][5] = 1
    
    col_offset = 0
    for r in range(7):
        A_map, B_map = terms[r]
        for l in range(7):
            global_col = col_offset + l
            
            # Update U
            for u_idx in range(4):
                if U2[u_idx][l] != 0:
                    val = U2[u_idx][l]
                    for g_idx, coeff in A_map[u_idx]:
                        U49[g_idx][global_col] += val * coeff
                        
            # Update V
            for v_idx in range(4):
                if V2[v_idx][l] != 0:
                    val = V2[v_idx][l]
                    for g_idx, coeff in B_map[v_idx]:
                        V49[g_idx][global_col] += val * coeff
                        
            # Update W
            for w_idx in range(4):
                if W2[w_idx][l] != 0:
                    val = W2[w_idx][l]
                    # w_idx corresponds to local k*2+i
                    local_k = w_idx // 2
                    local_i = w_idx % 2
                    for bi in range(2):
                        for bk in range(2):
                            coeff_c = c[bi][bk][r]
                            if coeff_c != 0:
                                global_i = bi * 2 + local_i
                                global_k = bk * 2 + local_k
                                global_w_idx = global_k * 4 + global_i
                                W49[global_w_idx][global_col] += val * coeff_c
                                
            col_offset += 1
            
    # Now construct 4x4x8 decomposition
    # Rank 98
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First half: B cols 0-3, C cols 0-3
    for i in range(16):
        for j in range(49):
            U[i][j] = U49[i][j]
            V[i][j] = V49[i][j]
            W[i][j] = W49[i][j]
            
    # Second half: B cols 4-7, C cols 4-7
    # U is same
    # V indices shifted by 16
    # W indices shifted by 16
    for i in range(16):
        for j in range(49):
            U[i][j+49] = U49[i][j]
            V[i+16][j+49] = V49[i][j]
            W[i+16][j+49] = W49[i][j]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
