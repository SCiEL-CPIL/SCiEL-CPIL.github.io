def mm_3x3_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
    mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2) # 17 terms
    for i in [i0, i1, i2]:
        for k in [k0, k1, k2]:
            add_term(...) # 9 terms
    return
