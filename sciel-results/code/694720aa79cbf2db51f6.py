import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 49
    # Sub-decompositions:
    # 1. Strassen 2x2x2 (Rank 7)
    # 2. Schoolbook 2x3x2 (Rank 12)
    # 3. Schoolbook 2x2x3 (Rank 12)
    # 4. Schoolbook 2x3x3 (Rank 18)
    
    rank_total = 7 + 12 + 12 + 18
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Helper to create schoolbook factors for n_sub x m_sub x p_sub
    def make_schoolbook(n_s, m_s, p_s):
        rank = n_s * m_s * p_s
        U_s = np.zeros((n_s * m_s, rank))
        V_s = np.zeros((m_s * p_s, rank))
        W_s = np.zeros((p_s * n_s, rank))
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U_s[i * m_s + j, col] = 1
                    V_s[j * p_s + k, col] = 1
                    W_s[k * n_s + i, col] = 1
                    col += 1
        return U_s, V_s, W_s

    # 1. Strassen 2x2x2
    # U_str (4x7), V_str (4x7), W_str (4x7)
    # Indices: u=i*2+j, v=j*2+k, w=k*2+i
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[1, 0] = 1; W_str[2, 0] = 1; W_str[3, 0] = 1 # Wait, W is k*n+i.
    # C00 += M1, C11 += M1
    # C00 is w=0 (k=0,i=0). C11 is w=3 (k=1,i=1).
    # My W_str assignment above is wrong.
    # Let's define W_str carefully.
    # W_str[w, r] where w = k*2+i.
    
    # Reset W_str
    W_str = np.zeros((4, 7))
    
    # M1: C00 += M1, C11 += M1
    W_str[0, 0] = 1
    W_str[3, 0] = 1
    
    # M2: C10 += M2, C11 -= M2
    # C10 is w=2 (k=1,i=0). C11 is w=3.
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[2, 1] = 1
    W_str[3, 1] = -1
    
    # M3: C01 += M3, C11 += M3
    # C01 is w=1 (k=0,i=1). C11 is w=3.
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[1, 2] = 1
    W_str[3, 2] = 1
    
    # M4: C00 += M4, C10 += M4
    # C00 is w=0. C10 is w=2.
    U_str[3, 3] = 1
    V_str[2, 3] = 1; V_str[0, 3] = -1
    W_str[0, 3] = 1
    W_str[2, 3] = 1
    
    # M5: C00 -= M5, C01 += M5
    # C00 is w=0. C01 is w=1.
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[0, 4] = -1
    W_str[1, 4] = 1
    
    # M6: C11 += M6
    # C11 is w=3.
    U_str[2, 5] = 1; U_str[0, 5] = -1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1
    
    # M7: C00 += M7
    # C00 is w=0.
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1
    
    # Embed Strassen into Global
    # U_str (4x7) -> U (10x7)
    # u_loc = i*2+j -> u_glob = i*5+j
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_loc = i * 2 + j
                u_glob = i * 5 + j
                U[u_glob, r] = U_str[u_loc, r]
                
    # V_str (4x7) -> V (25x7)
    # v_loc = j*2+k -> v_glob = j*5+k
    for r in range(7):
        for j in range(2):
            for k in range(2):
                v_loc = j * 2 + k
                v_glob = j * 5 + k
                V[v_glob, r] = V_str[v_loc, r]
                
    # W_str (4x7) -> W (10x7)
    # w_loc = k*2+i -> w_glob = k*2+i (since n=2, and k is 0,1)
    for r in range(7):
        for k in range(2):
            for i in range(2):
                w_loc = k * 2 + i
                w_glob = k * 2 + i
                W[w_glob, r] += W_str[w_loc, r]

    # 2. Schoolbook 2x3x2 (A3 B32)
    # A3: 2x3, B32: 3x2, C12: 2x2
    U_sc1, V_sc1, W_sc1 = make_schoolbook(2, 3, 2)
    offset = 7
    for r in range(12):
        # U_sc1 (6x12) -> U
        # u_loc = i*3+j'' -> u_glob = i*5+(2+j'')
        for i in range(2):
            for j_p in range(3):
                u_loc = i * 3 + j_p
                u_glob = i * 5 + (2 + j_p)
                U[u_glob, offset + r] = U_sc1[u_loc, r]
                
        # V_sc1 (6x12) -> V
        # v_loc = j''*2+k -> v_glob = (2+j'')*5+k
        for j_p in range(3):
            for k in range(2):
                v_loc = j_p * 2 + k
                v_glob = (2 + j_p) * 5 + k
                V[v_glob, offset + r] = V_sc1[v_loc, r]
                
        # W_sc1 (4x12) -> W
        # w_loc = k*2+i -> w_glob = k*2+i
        for k in range(2):
            for i in range(2):
                w_loc = k * 2 + i
                w_glob = k * 2 + i
                W[w_glob, offset + r] += W_sc1[w_loc, r]

    # 3. Schoolbook 2x2x3 (A2 B23)
    # A2: 2x2, B23: 2x3, C21: 2x3
    U_sc2, V_sc2, W_sc2 = make_schoolbook(2, 2, 3)
    offset = 7 + 12
    for r in range(12):
        # U_sc2 (4x12) -> U
        # u_loc = i*2+j -> u_glob = i*5+j
        for i in range(2):
            for j in range(2):
                u_loc = i * 2 + j
                u_glob = i * 5 + j
                U[u_glob, offset + r] = U_sc2[u_loc, r]
                
        # V_sc2 (6x12) -> V
        # v_loc = j*3+k'' -> v_glob = j*5+(2+k'')
        for j in range(2):
            for k_p in range(3):
                v_loc = j * 3 + k_p
                v_glob = j * 5 + (2 + k_p)
                V[v_glob, offset + r] = V_sc2[v_loc, r]
                
        # W_sc2 (6x12) -> W
        # w_loc = k''*2+i -> w_glob = (2+k'')*2+i
        for k_p in range(3):
            for i in range(2):
                w_loc = k_p * 2 + i
                w_glob = (2 + k_p) * 2 + i
                W[w_glob, offset + r] += W_sc2[w_loc, r]

    # 4. Schoolbook 2x3x3 (A3 B33)
    # A3: 2x3, B33: 3x3, C22: 2x3
    U_sc3, V_sc3, W_sc3 = make_schoolbook(2, 3, 3)
    offset = 7 + 12 + 12
    for r in range(18):
        # U_sc3 (6x18) -> U
        # u_loc = i*3+j'' -> u_glob = i*5+(2+j'')
        for i in range(2):
            for j_p in range(3):
                u_loc = i * 3 + j_p
                u_glob = i * 5 + (2 + j_p)
                U[u_glob, offset + r] = U_sc3[u_loc, r]
                
        # V_sc3 (9x18) -> V
        # v_loc = j''*3+k'' -> v_glob = (2+j'')*5+(2+k'')
        for j_p in range(3):
            for k_p in range(3):
                v_loc = j_p * 3 + k_p
                v_glob = (2 + j_p) * 5 + (2 + k_p)
                V[v_glob, offset + r] = V_sc3[v_loc, r]
                
        # W_sc3 (6x18) -> W
        # w_loc = k''*2+i -> w_glob = (2+k'')*2+i
        for k_p in range(3):
            for i in range(2):
                w_loc = k_p * 2 + i
                w_glob = (2 + k_p) * 2 + i
                W[w_glob, offset + r] += W_sc3[w_loc, r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
