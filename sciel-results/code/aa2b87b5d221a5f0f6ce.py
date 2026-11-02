for (li, lj), coeff_u in u_dict.items():
    gi, gj = i_base + li, j_base + lj
    U[gi * m + gj][col_start] += coeff_u
