"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_val, v_val, w_val):
        U_rows.append(u_val)
        V_rows.append(v_val)
        W_rows.append(w_val)
    
    def vec(length, pos):
        v = [0] * length
        v[pos] = 1
        return v
    
    def vec_multi(length, pos_val_pairs):
        v = [0] * length
        for pos, val in pos_val_pairs:
            v[pos] += val
        return v
    
    def strassen_terms_explicit(p_idx, q_idx, r_idx, u_len, v_len, w_len):
        p00, p01, p10, p11 = p_idx
        q00, q01, q10, q11 = q_idx
        r00, r01, r10, r11 = r_idx
        
        u = vec_multi(u_len, [(p00, 1), (p11, 1)])
        v = vec_multi(v_len, [(q00, 1), (q11, 1)])
        w = vec_multi(w_len, [(r00, 1), (r01, -1), (r10, 1), (r11, 1)])
        add_term(u, v, w)
        
        u = vec_multi(u_len, [(p10, 1), (p11, 1)])
        v = vec_multi(v_len, [(q00, 1)])
        w = vec_multi(w_len, [(r00, -1), (r10, 1), (r11, -1)])
        add_term(u, v, w)
        
        u = vec_multi(u_len, [(p00, 1)])
        v = vec_multi(v_len, [(q01, 1), (q11, -1)])
        w = vec_multi(w_len, [(r00, 1), (r01, 1), (r11, 1)])
        add_term(u, v, w)
        
        u = vec_multi(u_len, [(p11, 1)])
        v = vec_multi(v_len, [(q10, 1), (q00, -1)])
        w = vec_multi(w_len, [(r00, 1), (r01, 1)])
        add_term(u, v, w)
        
        u = vec_multi(u_len, [(p00, 1), (p01, 1)])
        v = vec_multi(v_len, [(q11, 1)])
        w = vec_multi(w_len, [(r01, 1), (r11, 1)])
        add_term(u, v, w)
        
        u = vec_multi(u_len, [(p10, 1), (p00, -1)])
        v = vec_multi(v_len, [(q00, 1), (q01, 1)])
        w = vec_multi(w_len, [(r01, -1)])
        add_term(u, v, w)
        
        u = vec_multi(u_len, [(p11, 1), (p01, -1)])
        v = vec_multi(v_len, [(q10, 1), (q11, 1)])
        w = vec_multi(w_len, [(r01, -1), (r11, 1)])
        add_term(u, v, w)
    
    def schoolbook_2x2x1_explicit(p_idx, q_idx, r_idx, u_len, v_len, w_len):
        p00, p01, p10, p11 = p_idx
        q0, q1 = q_idx
        r0, r1 = r_idx
        
        u = vec(u_len, p00); v = vec(v_len, q0); w = vec(w_len, r0); add_term(u, v, w)
        u = vec(u_len, p01); v = vec(v_len, q1); w = vec(w_len, r0); add_term(u, v, w)
        u = vec(u_len, p10); v = vec(v_len, q0); w = vec(w_len, r1); add_term(u, v, w)
        u = vec(u_len, p11); v = vec(v_len, q1); w = vec(w_len, r1); add_term(u, v, w)
    
    u_len = 8
    v_len = 28
    w_len = 14
    
    a1_idx = [0, 1, 4, 5]
    a2_idx = [2, 3, 6, 7]
    
    def process_2x2x7(a_idx, b_row0, b_row1):
        # Cols 0,1: Strassen
        q_idx = [b_row0[0], b_row0[1], b_row1[0], b_row1[1]]
        r_idx = [0, 2, 1, 3]  # C[0,0],C[0,1],C[1,0],C[1,1] -> k*2+i
        strassen_terms_explicit(a_idx, q_idx, r_idx, u_len, v_len, w_len)
        
        # Col 2: schoolbook
        q_idx1 = [b_row0[2], b_row1[2]]
        r_idx1 = [4, 5]  # C[0,2],C[1,2]
        schoolbook_2x2x1_explicit(a_idx, q_idx1, r_idx1, u_len, v_len, w_len)
        
        # Cols 3,4: Strassen
        q_idx = [b_row0[3], b_row0[4], b_row1[3], b_row1[4]]
        r_idx = [6, 8, 7, 9]
        strassen_terms_explicit(a_idx, q_idx, r_idx, u_len, v_len, w_len)
        
        # Cols 5,6: Strassen
        q_idx = [b_row0[5], b_row0[6], b_row1[5], b_row1[6]]
        r_idx = [10, 12, 11, 13]
        strassen_terms_explicit(a_idx, q_idx, r_idx, u_len, v_len, w_len)
    
    b1_row0 = list(range(7))
    b1_row1 = list(range(7, 14))
    b2_row0 = list(range(14, 21))
    b2_row1 = list(range(21, 28))
    
    process_2x2x7(a1_idx, b1_row0, b1_row1)
    process_2x2x7(a2_idx, b2_row0, b2_row1)
    
    rank = len(U_rows)
    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
