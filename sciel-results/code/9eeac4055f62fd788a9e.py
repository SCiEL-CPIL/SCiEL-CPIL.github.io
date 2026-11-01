import time

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 25
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Strassen blocks for B columns 0-1, 2-3, 4-5
    for b in range(3):
        col_offset = b * 7
        # B indices for this block:
        # x00: 0*7 + 2*b
        # x01: 0*7 + 2*b + 1
        # x10: 1*7 + 2*b
        # x11: 1*7 + 2*b + 1
        v_x00 = 0 * p + 2 * b
        v_x01 = 0 * p + 2 * b + 1
        v_x10 = 1 * p + 2 * b
        v_x11 = 1 * p + 2 * b + 1
        
        # W indices for this block:
        # c00: k=2b, i=0 -> 2b*n + 0
        # c01: k=2b+1, i=0 -> (2b+1)*n + 0
        # c10: k=2b, i=1 -> 2b*n + 1
        # c11: k=2b+1, i=1 -> (2b+1)*n + 1
        w_c00 = 2 * b * n + 0
        w_c01 = (2 * b + 1) * n + 0
        w_c10 = 2 * b * n + 1
        w_c11 = (2 * b + 1) * n + 1
        
        # U indices are fixed:
        # a00: 0, a01: 1, a10: 2, a11: 3
        
        # M1: (a00+a11)(x00+x11)
        c = col_offset + 0
        add_term(c, 0, 1, v_x00, 1, w_c00, 1)
        add_term(c, 3, 1, v_x11, 1, w_c11, 1)
        
        # M2: (a10+a11)x00
        c = col_offset + 1
        add_term(c, 2, 1, v_x00, 1, w_c10, 1)
        add_term(c, 3, 1, v_x00, 1, w_c11, -1)
        
        # M3: a00(x10-x11)
        c = col_offset + 2
        add_term(c, 0, 1, v_x10, 1, w_c01, 1)
        add_term(c, 0, 1, v_x11, -1, w_c11, 1)
        
        # M4: a11(x01-x00)
        c = col_offset + 3
        add_term(c, 3, 1, v_x01, 1, w_c00, 1)
        add_term(c, 3, 1, v_x00, -1, w_c10, 1)
        
        # M5: (a00+a01)x11
        c = col_offset + 4
        add_term(c, 0, 1, v_x11, 1, w_c00, -1)
        add_term(c, 1, 1, v_x11, 1, w_c01, 1)
        
        # M6: (a10-a00)(x00+x01)
        c = col_offset + 5
        add_term(c, 2, 1, v_x00, 1, w_c11, 1)
        add_term(c, 0, -1, v_x00, 1, w_c11, 1)
        add_term(c, 2, 1, v_x01, 1, w_c11, 1)
        add_term(c, 0, -1, v_x01, 1, w_c11, 1)
        
        # M7: (a01-a11)(x10+x11)
        c = col_offset + 6
        add_term(c, 1, 1, v_x10, 1, w_c00, 1)
        add_term(c, 3, -1, v_x10, 1, w_c00, 1)
        add_term(c, 1, 1, v_x11, 1, w_c00, 1)
        add_term(c, 3, -1, v_x11, 1, w_c00, 1)

    # Schoolbook for B column 6
    col_offset = 21
    v_y00 = 0 * p + 6
    v_y10 = 1 * p + 6
    w_c06 = 6 * n + 0
    w_c16 = 6 * n + 1
    
    # N1: a00 y00
    c = col_offset + 0
    add_term(c, 0, 1, v_y00, 1, w_c06, 1)
    
    # N2: a01 y10
    c = col_offset + 1
    add_term(c, 1, 1, v_y10, 1, w_c06, 1)
    
    # N3: a10 y00
    c = col_offset + 2
    add_term(c, 2, 1, v_y00, 1, w_c16, 1)
    
    # N4: a11 y10
    c = col_offset + 3
    add_term(c, 3, 1, v_y10, 1, w_c16, 1)

    return {"U": U, "V": V, "W": W}
