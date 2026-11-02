"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    def get_strassen_2x2():
        # Returns list of (U, V, W) tuples for 2x2x2
        # U, V, W are lists of length 4
        terms = []
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1]
        terms.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1]
        terms.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1]
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0]
        terms.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0]
        terms.append((u, v, w))
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1]
        terms.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0]
        terms.append((u, v, w))
        
        return terms

    def construct_4x4x4():
        s2 = get_strassen_2x2()
        terms = []
        for i_out in range(7):
            u_out, v_out, w_out = s2[i_out]
            for i_in in range(7):
                u_in, v_in, w_in = s2[i_in]
                
                u_4x4 = [0] * 16
                v_4x4 = [0] * 16
                w_4x4 = [0] * 16
                
                for I in range(2):
                    for J in range(2):
                        val_u_out = u_out[I*2+J]
                        if val_u_out != 0:
                            for i in range(2):
                                for j in range(2):
                                    val_u_in = u_in[i*2+j]
                                    if val_u_in != 0:
                                        global_i = 2*I + i
                                        global_j = 2*J + j
                                        u_4x4[global_i*4 + global_j] += val_u_out * val_u_in
                                        
                for J in range(2):
                    for K in range(2):
                        val_v_out = v_out[J*2+K]
                        if val_v_out != 0:
                            for j in range(2):
                                for k in range(2):
                                    val_v_in = v_in[j*2+k]
                                    if val_v_in != 0:
                                        global_j = 2*J + j
                                        global_k = 2*K + k
                                        v_4x4[global_j*4 + global_k] += val_v_out * val_v_in
                                        
                for K in range(2):
                    for I in range(2):
                        val_w_out = w_out[K*2+I]
                        if val_w_out != 0:
                            for k in range(2):
                                for i in range(2):
                                    val_w_in = w_in[k*2+i]
                                    if val_w_in != 0:
                                        global_k = 2*K + k
                                        global_i = 2*I + i
                                        w_4x4[global_k*4 + global_i] += val_w_out * val_w_in
                                        
                terms.append((u_4x4, v_4x4, w_4x4))
        return terms

    terms_4x4 = construct_4x4x4()
    
    U_list = []
    V_list = []
    W_list = []
    
    for u, v, w in terms_4x4:
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    for u, v, w in terms_4x4:
        U_list.append(u)
        v_new = [0]*16
        for j in range(4):
            for k in range(4):
                v_new[j*4 + (k+4)] = v[j*4+k]
        V_list.append(v_new)
        
        w_new = [0]*16
        for k in range(4):
            for i in range(4):
                w_new[(k+4)*4 + i] = w[k*4+i]
        W_list.append(w_new)
        
    U_final = []
    V_final = []
    W_final = []
    
    for r in range(len(U_list)):
        U_final.append(U_list[r])
        V_final.append(V_list[r] + [0]*16)
        W_final.append(W_list[r] + [0]*16)
        
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
