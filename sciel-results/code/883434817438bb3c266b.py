import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {local_index: coeff}
    # local index for 2x2 matrix [a b; c d] is a=0, b=1, c=2, d=3
    
    strassen_terms = [
        # P1: (x00+x11)(y00+y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (x10+x11)y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: x00(y01-y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: x11(y10-y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5: (x00+x01)y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6: (x10-x00)(y00+y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7: (x01-x11)(y10+y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_block(col_offset, u_map, v_map, w_map):
        """
        u_map: dict {local_idx: global_idx}
        v_map: dict {local_idx: global_idx}
        w_map: dict {local_idx: global_idx}
        """
        for t in range(7):
            u_c, v_c, w_c = strassen_terms[t]
            for local_u, coeff_u in u_c.items():
                global_u = u_map[local_u]
                U[global_u][col_offset + t] += coeff_u
            for local_v, coeff_v in v_c.items():
                global_v = v_map[local_v]
                V[global_v][col_offset + t] += coeff_v
            for local_w, coeff_w in w_c.items():
                global_w = w_map[local_w]
                W[global_w][col_offset + t] += coeff_w
                
    # Define index maps for the 4 sub-products
    # Local indices: 0,1,2,3 correspond to [0,0], [0,1], [1,0], [1,1]
    
    # 1. A_L * B_LL -> C_L
    # A_L: A[0:2, 0:2] -> U indices: 0, 1, 4, 5
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    # B_LL: B[0:2, 0:2] -> V indices: 0, 1, 4, 5
    v_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    # C_L: C[0:2, 0:2] -> W indices:
    # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    # Local 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
    w_map_1 = {0: 0, 1: 2, 2: 1, 3: 3}
    
    # 2. A_R * B_RL -> C_L
    # A_R: A[0:2, 2:4] -> U indices: 2, 3, 6, 7
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7}
    # B_RL: B[2:4, 0:2] -> V indices: 8, 9, 12, 13
    v_map_2 = {0: 8, 1: 9, 2: 12, 3: 13}
    # C_L: Same as above
    w_map_2 = {0: 0, 1: 2, 2: 1, 3: 3}
    
    # 3. A_L * B_LR -> C_R
    # A_L: Same as above
    u_map_3 = {0: 0, 1: 1, 2: 4, 3: 5}
    # B_LR: B[0:2, 2:4] -> V indices: 2, 3, 6, 7
    v_map_3 = {0: 2, 1: 3, 2: 6, 3: 7}
    # C_R: C[0:2, 2:4] -> W indices:
    # C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    # Local 0->C[0,2], 1->C[0,3], 2->C[1,2], 3->C[1,3]
    w_map_3 = {0: 4, 1: 6, 2: 5, 3: 7}
    
    # 4. A_R * B_RR -> C_R
    # A_R: Same as above
    u_map_4 = {0: 2, 1: 3, 2: 6, 3: 7}
    # B_RR: B[2:4, 2:4] -> V indices: 10, 11, 14, 15
    v_map_4 = {0: 10, 1: 11, 2: 14, 3: 15}
    # C_R: Same as above
    w_map_4 = {0: 4, 1: 6, 2: 5, 3: 7}
    
    add_block(0, u_map_1, v_map_1, w_map_1)
    add_block(7, u_map_2, v_map_2, w_map_2)
    add_block(14, u_map_3, v_map_3, w_map_3)
    add_block(21, u_map_4, v_map_4, w_map_4)
    
    return {"U": U, "V": V, "W": W}
