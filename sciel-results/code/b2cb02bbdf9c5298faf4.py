def strassen_2x2(U, V, W, rank_offset, a_row, a_col, b_row, b_col, c_row, c_col):
    # M1
    r = rank_offset
    # A: (0,0), (1,1)
    U[a_row*4 + a_col, r] += 1
    U[(a_row+1)*4 + (a_col+1), r] += 1
    # B: (0,0), (1,1)
    V[b_row*6 + b_col, r] += 1
    V[(b_row+1)*6 + (b_col+1), r] += 1
    # C: (0,0), (1,1)
    W[c_col*2 + c_row, r] += 1
    W[(c_col+1)*2 + (c_row+1), r] += 1
    # ... and so on for M2..M7
