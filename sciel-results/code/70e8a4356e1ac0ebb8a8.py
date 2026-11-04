import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 19
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u, v, w):
        nonlocal col
        for i in range(len(u)):
            if u[i] != 0:
                U[i][col] = u[i]
        for i in range(len(v)):
            if v[i] != 0:
                V[i][col] = v[i]
        for i in range(len(w)):
            if w[i] != 0:
                W[i][col] = w[i]
        col += 1

    # Sub-problem 1: 2x2x2 (Strassen)
    # i in 0,1; j in 0,1; k in 0,1
    # Local U index: i*2+j
    # Local V index: j*2+k
    # Local W index: k*2+i
    
    # Strassen factors for 2x2
    # M1 = (A00+A11)(B00+B11)
    u1 = [1, 0, 0, 1] # A00, A01, A10, A11
    v1 = [1, 0, 1, 0] # B00, B01, B10, B11
    w1 = [1, 0, 0, 1] # C00, C01, C10, C11 -> M1 contributes to C00(+1), C11(+1)
    # Wait, W index is k*n+i.
    # C00: k=0, i=0 -> 0
    # C11: k=1, i=1 -> 1*2+1=3
    # So w1[0]=1, w1[3]=1.
    w1 = [1, 0, 0, 1]
    
    # Map local to global
    def map_u_local(u_local, j_offset=0):
        u_global = [0] * (n*m)
        for i in range(2):
            for j in range(2):
                idx_local = i*2 + j
                idx_global = i*m + (j + j_offset)
                u_global[idx_global] = u_local[idx_local]
        return u_global

    def map_v_local(v_local, j_offset=0, k_offset=0):
        v_global = [0] * (m*p)
        for j in range(2):
            for k in range(2):
                idx_local = j*2 + k
                idx_global = (j + j_offset)*p + (k + k_offset)
                v_global[idx_global] = v_local[idx_local]
        return v_global

    def map_w_local(w_local, j_offset=0, k_offset=0):
        w_global = [0] * (p*n)
        for k in range(2):
            for i in range(2):
                idx_local = k*2 + i
                idx_global = (k + k_offset)*n + i
                w_global[idx_global] = w_local[idx_local]
        return w_global

    # Strassen 7 terms
    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 1, 0]
    w1 = [1, 0, 0, 1]
    add_term(map_u_local(u1), map_v_local(v1, 0, 0), map_w_local(w1, 0, 0))
    
    # M2 = (A10+A11)B00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0] # C10: k=0, i=1 -> 1
    add_term(map_u_local(u2), map_v_local(v2, 0, 0), map_w_local(w2, 0, 0))
    
    # M3 = A00(B01-B11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 0] # C01: k=1, i=0 -> 2
    add_term(map_u_local(u3), map_v_local(v3, 0, 0), map_w_local(w3, 0, 0))
    
    # M4 = A11(B10-B00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 0, 0] # C00: k=0, i=0 -> 0? No.
    # M4 contributes to C00(-1) and C10(+1).
    # C00: k=0, i=0 -> 0
    # C10: k=0, i=1 -> 1
    w4 = [-1, 0, 1, 0]
    add_term(map_u_local(u4), map_v_local(v4, 0, 0), map_w_local(w4, 0, 0))
    
    # M5 = (A00+A01)B11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 1, 0, 0] # C01: k=1, i=0 -> 2?
    # M5 contributes to C01(+1) and C00(-1).
    # C01: k=1, i=0 -> 2
    # C00: k=0, i=0 -> 0
    w5 = [-1, 0, 0, 1]
    add_term(map_u_local(u5), map_v_local(v5, 0, 0), map_w_local(w5, 0, 0))
    
    # M6 = (A10-A00)(B00+B01)
    u6 = [0, 0, 1, -1]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11: k=1, i=1 -> 3
    # M6 contributes to C11(+1).
    w6 = [0, 0, 0, 1]
    add_term(map_u_local(u6), map_v_local(v6, 0, 0), map_w_local(w6, 0, 0))
    
    # M7 = (A01-A11)(B10+B11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [0, 0, 0, 1] # C11: k=1, i=1 -> 3?
    # M7 contributes to C11(+1) and C00(+1).
    # C11: k=1, i=1 -> 3
    # C00: k=0, i=0 -> 0
    w7 = [1, 0, 0, 1]
    add_term(map_u_local(u7), map_v_local(v7, 0, 0), map_w_local(w7, 0, 0))
    
    # Sub-problem 2: 2x2x2 (Strassen)
    # j in 0,1; k in 2,3
    # Same U, V, W but with k_offset=2
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 1, 0]
    w1 = [1, 0, 0, 1]
    add_term(map_u_local(u1), map_v_local(v1, 0, 2), map_w_local(w1, 0, 2))
    
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    add_term(map_u_local(u2), map_v_local(v2, 0, 2), map_w_local(w2, 0, 2))
    
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 0]
    add_term(map_u_local(u3), map_v_local(v3, 0, 2), map_w_local(w3, 0, 2))
    
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [-1, 0, 1, 0]
    add_term(map_u_local(u4), map_v_local(v4, 0, 2), map_w_local(w4, 0, 2))
    
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 0, 0, 1]
    add_term(map_u_local(u5), map_v_local(v5, 0, 2), map_w_local(w5, 0, 2))
    
    u6 = [0, 0, 1, -1]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    add_term(map_u_local(u6), map_v_local(v6, 0, 2), map_w_local(w6, 0, 2))
    
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 1]
    add_term(map_u_local(u7), map_v_local(v7, 0, 2), map_w_local(w7, 0, 2))
    
    # Sub-problem 3: 2x2x1 (Schoolbook)
    # j in 0,1; k in 4
    for i in range(2):
        for j in range(2):
            u = [0] * (n*m)
            v = [0] * (m*p)
            w = [0] * (p*n)
            u[i*m + j] = 1
            v[j*p + 4] = 1
            w[4*n + i] = 1
            add_term(u, v, w)
            
    # Sub-problem 4: 2x1x5 (Rank 1)
    # j in 2; k in 0..4
    u = [0] * (n*m)
    v = [0] * (m*p)
    w = [0] * (p*n)
    for i in range(2):
        u[i*m + 2] = 1
    for k in range(5):
        v[2*p + k] = 1
        for i in range(2):
            w[k*n + i] = 1
    add_term(u, v, w)
    
    return {"U": U, "V": V, "W": W}
