<<<<<<< SEARCH
    # 6 Strassen blocks for k0 = 0,2,4,6,8,10
    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    # 1 single block for k=12
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={R}"
=======
    # 13 single blocks for k = 0..12
    col = 0
    for k in range(13):
        add_single_block(k, col)
        col += 4
    assert col == R, f"col={col}, R={R}"
>>>>>>> REPLACE
