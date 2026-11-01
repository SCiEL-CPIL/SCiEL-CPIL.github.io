def add_strassen_block(a_indices, b_indices, c_indices):
    """
    a_indices: [a00, a01, a10, a11]
    b_indices: [b00, b01, b10, b11]
    c_indices: [c00, c01, c10, c11] corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    """
    a00, a01, a10, a11 = a_indices
    b00, b01, b10, b11 = b_indices
    c00, c01, c10, c11 = c_indices
    
    # m1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    add_rank_step([(a00, 1), (a11, 1)], 
                  [(b00, 1), (b11, 1)], 
                  [(c00, 1), (c11, 1)])
    
    # m2 = (a10+a11)*b00 -> C10 += 1, C11 -= 1
    add_rank_step([(a10, 1), (a11, 1)], 
                  [(b00, 1)], 
                  [(c10, 1), (c11, -1)])
    
    # m3 = a00*(b01-b11) -> C01 += 1, C11 += 1
    add_rank_step([(a00, 1)], 
                  [(b01, 1), (b11, -1)], 
                  [(c01, 1), (c11, 1)])
    
    # m4 = a11*(b10-b00) -> C00 += 1, C10 += 1
    add_rank_step([(a11, 1)], 
                  [(b10, 1), (b00, -1)], 
                  [(c00, 1), (c10, 1)])
    
    # m5 = (a00+a01)*b11 -> C00 -= 1, C01 += 1
    add_rank_step([(a00, 1), (a01, 1)], 
                  [(b11, 1)], 
                  [(c00, -1), (c01, 1)])
    
    # m6 = (a10-a00)*(b00+b01) -> C11 += 1
    add_rank_step([(a10, 1), (a00, -1)], 
                  [(b00, 1), (b01, 1)], 
                  [(c11, 1)])
    
    # m7 = (a01-a11)*(b10+b11) -> C00 += 1
    add_rank_step([(a01, 1), (a11, -1)], 
                  [(b10, 1), (b11, 1)], 
                  [(c00, 1)])
