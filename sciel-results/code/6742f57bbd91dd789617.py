import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Pre-compute Strassen 2x2x2
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U_s[0][0]=1; U_s[3][0]=1
    V_s[0][0]=1; V_s[3][0]=1
    W_s[0][0]=1; W_s[3][0]=1 # C00, C11
    # Wait, W indices are for C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify Strassen W matrix carefully.
    # C00: M1, M4, -M5, M7
    W_s[0][0]=1; W_s[0][3]=1; W_s[0][4]=-1; W_s[0][6]=1
    # C01: M3, M5
    W_s[1][2]=1; W_s[1][4]=1
    # C10: M2, M4
    W_s[2][1]=1; W_s[2][3]=1
    # C11: M1, -M2, M3, M6
    W_s[3][0]=1; W_s[3][1]=-1; W_s[3][2]=1; W_s[3][5]=1
    
    # M2: (A10+A11)B00
    U_s[2][1]=1; U_s[3][1]=1
    V_s[0][1]=1
    # M3: A00(B10-B11)
    U_s[0][2]=1
    V_s[2][2]=1; V_s[3][2]=-1
    # M4: A11(B01-B00)
    U_s[3][3]=1
    V_s[1][3]=1; V_s[0][3]=-1
    # M5: (A00+A01)B11
    U_s[0][4]=1; U_s[1][4]=1
    V_s[3][4]=1
    # M6: (A10-A00)(B00+B01)
    U_s[2][5]=1; U_s[0][5]=-1
    V_s[0][5]=1; V_s[1][5]=1
    # M7: (A01-A11)(B10+B11)
    U_s[1][6]=1; U_s[3][6]=-1
    V_s[2][6]=1; V_s[3][6]=1
    
    # Pre-compute Schoolbook 2x1x2
    U_sb = [[0]*4 for _ in range(2)]
    V_sb = [[0]*4 for _ in range(2)] # 1x2 -> 2 entries
    W_sb = [[0]*4 for _ in range(4)] # 2x2 -> 4 entries
    
    # C00 = A0 B0
    U_sb[0][0]=1; V_sb[0][0]=1; W_sb[0][0]=1
    # C01 = A0 B1
    U_sb[0][1]=1; V_sb[1][1]=1; W_sb[1][1]=1
    # C10 = A1 B0
    U_sb[1][2]=1; V_sb[0][2]=1; W_sb[2][2]=1
    # C11 = A1 B1
    U_sb[1][3]=1; V_sb[1][3]=1; W_sb[3][3]=1
    
    # Global dimensions
    R = 108
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    rank_offset = 0
    
    # Split indices
    # I: 0,1 (size 2), 2,3 (size 2)
    # J: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    # K: 0,1 (size 2), 2,3 (size 2), 4,5 (size 2)
    
    I_blocks = [(0, 2), (2, 4)] # start, end
    J_blocks = [(0, 2), (2, 4), (4, 5)]
    K_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for i_start, i_end in I_blocks:
        i_size = i_end - i_start
        for j_start, j_end in J_blocks:
            j_size = j_end - j_start
            for k_start, k_end in K_blocks:
                k_size = k_end - k_start
                
                if j_size == 2:
                    # Use Strassen
                    # U_s: 4x7, V_s: 4x7, W_s: 4x7
                    # Local A index: (i_local, j_local) -> i_local*2 + j_local
                    # Local B index: (j_local, k_local) -> j_local*2 + k_local
                    # Local C index: (i_local, k_local) -> i_local*2 + k_local
                    
                    for r in range(7):
                        # Fill U
                        for u_local in range(4):
                            val = U_s[u_local][r]
                            if val != 0:
                                i_local = u_local // 2
                                j_local = u_local % 2
                                i_global = i_start + i_local
                                j_global = j_start + j_local
                                U[i_global*m + j_global][rank_offset + r] = val
                                
                        # Fill V
                        for v_local in range(4):
                            val = V_s[v_local][r]
                            if val != 0:
                                j_local = v_local // 2
                                k_local = v_local % 2
                                j_global = j_start + j_local
                                k_global = k_start + k_local
                                V[j_global*p + k_global][rank_offset + r] = val
                                
                        # Fill W
                        for w_local in range(4):
                            val = W_s[w_local][r]
                            if val != 0:
                                i_local = w_local // 2
                                k_local = w_local % 2
                                i_global = i_start + i_local
                                k_global = k_start + k_local
                                W[k_global*n + i_global][rank_offset + r] = val
                                
                    rank_offset += 7
                else:
                    # j_size == 1
                    # Use Schoolbook 2x1x2
                    # U_sb: 2x4, V_sb: 2x4, W_sb: 4x4
                    # Local A index: i_local (0..1)
                    # Local B index: k_local (0..1)
                    # Local C index: i_local*2 + k_local
                    
                    for r in range(4):
                        # Fill U
                        for u_local in range(2):
                            val = U_sb[u_local][r]
                            if val != 0:
                                i_global = i_start + u_local
                                j_global = j_start
                                U[i_global*m + j_global][rank_offset + r] = val
                                
                        # Fill V
                        for v_local in range(2):
                            val = V_sb[v_local][r]
                            if val != 0:
                                j_global = j_start
                                k_global = k_start + v_local
                                V[j_global*p + k_global][rank_offset + r] = val
                                
                        # Fill W
                        for w_local in range(4):
                            val = W_sb[w_local][r]
                            if val != 0:
                                i_local = w_local // 2
                                k_local = w_local % 2
                                i_global = i_start + i_local
                                k_global = k_start + k_local
                                W[k_global*n + i_global][rank_offset + r] = val
                                
                    rank_offset += 4
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
