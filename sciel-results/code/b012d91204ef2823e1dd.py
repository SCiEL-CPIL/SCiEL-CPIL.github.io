def set_W_block(row_offset, col_offset, val, col):
    """
    row_offset: offset for i (row in C)
    col_offset: offset for k (column in C)
    """
    for di in range(2):
        for dk in range(2):
            i = row_offset + di
            k = col_offset + dk
            W[k * 4 + i][col] = val
