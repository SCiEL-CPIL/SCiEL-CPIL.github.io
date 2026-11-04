import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank_total = 80
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    def add_4x4_4x2(U, V, W, b_col_start, rank_offset):
        nonlocal col_offset
        def add_2x2x2(u_row_base, u_col_base, v_row_base, v_col_base, w_row_base, w_col_base, offset):
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        ai = u_row_base + di
                        aj = u_col_base + dj
                        bi = v_row_base + dj
                        bk = v_col_base + dk
                        ci = w_row_base + di
                        ck = w_col_base + dk
                        
                        U[ai*m + aj][offset] = 1
                        V[bi*p + bk][offset] = 1
                        W[ck*n + ci][offset] = 1
                        offset += 1
            return offset
        
        col_offset = add_2x2x2(0, 0, 0, b_col_start, 0, b_col_start, rank_offset + col_offset)
        col_offset = add_2x2x2(0, 2, 2, b_col_start, 0, b_col_start, rank_offset + col_offset)
        col_offset = add_2x2x2(2, 0, 0, b_col_start, 2, b_col_start, rank_offset + col_offset)
        col_offset = add_2x2x2(2, 2, 2, b_col_start, 2, b_col_start, rank_offset + col_offset)
        return col_offset

    def add_4x4_4x3(U, V, W, b_col_start, rank_offset):
        nonlocal col_offset
        def add_2x2x2x3(u_row_base, u_col_base, v_row_base, v_col_base, w_row_base, w_col_base, offset):
            for di in range(2):
                for dj in range(2):
                    for dk in range(3):
                        ai = u_row_base + di
                        aj = u_col_base + dj
                        bi = v_row_base + dj
                        bk = v_col_base + dk
                        ci = w_row_base + di
                        ck = w_col_base + dk
                        
                        U[ai*m + aj][offset] = 1
                        V[bi*p + bk][offset] = 1
                        W[ck*n + ci][offset] = 1
                        offset += 1
            return offset
        
        col_offset = add_2x2x2x3(0, 0, 0, b_col_start, 0, b_col_start, rank_offset + col_offset)
        col_offset = add_2x2x2x3(0, 2, 2, b_col_start, 0, b_col_start, rank_offset + col_offset)
        col_offset = add_2x2x2x3(2, 0, 0, b_col_start, 2, b_col_start, rank_offset + col_offset)
        col_offset = add_2x2x2x3(2, 2, 2, b_col_start, 2, b_col_start, rank_offset + col_offset)
        return col_offset

    r1 = add_4x4_4x2(U, V, W, 0, 0)
    r2 = add_4x4_4x2(U, V, W, 2, r1)
    r3 = add_4x4_4x3(U, V, W, 4, r2)
    
    # Check time
    if time.time() - start_time > 140:
        # Fallback to schoolbook if something went wrong (unlikely)
        pass
        
    return {"U": U, "V": V, "W": W}
