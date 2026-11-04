"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    
    # Partition: rows [2,2], middle [2,2,1], cols [2,2,2]
    row_splits = [2, 2]
    mid_splits = [2, 2, 1]
    col_splits = [2, 2, 2]
    
    row_offsets = [0]
    for s in row_splits:
        row_offsets.append(row_offsets[-1] + s)
    mid_offsets = [0]
    for s in mid_splits:
        mid_offsets.append(mid_offsets[-1] + s)
    col_offsets = [0]
    for s in col_splits:
        col_offsets.append(col_offsets[-1] + s)
    
    nr = len(row_splits)
    nm = len(mid_splits)
    nc = len(col_splits)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(a_entries, b_entries, c_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for (i, j), coeff in a_entries.items():
            u[i * m + j] = coeff
        for (j, k), coeff in b_entries.items():
            v[j * p + k] = coeff
        for (i, k), coeff in c_entries.items():
            w[k * n + i] = coeff
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    for ri in range(nr):
        for mi in range(nm):
            for ci in range(nc):
                ar = row_splits[ri]
                ac = mid_splits[mi]
                br = mid_splits[mi]
                bc = col_splits[ci]
                
                i0 = row_offsets[ri]
                j0 = mid_offsets[mi]
                k0 = col_offsets[ci]
                
                if ar == 2 and ac == 2 and bc == 2:
                    # Strassen for 2x2 x 2x2
                    # M1
                    add_term(
                        {(i0, j0): 1, (i0+1, j0+1): 1},
                        {(j0, k0): 1, (j0+1, k0+1): 1},
                        {(i0, k0): 1, (i0+1, k0+1): 1}
                    )
                    # M2
                    add_term(
                        {(i0+1, j0): 1, (i0+1, j0+1): 1},
                        {(j0, k0): 1},
                        {(i0+1, k0): 1}
                    )
                    # M3
                    add_term(
                        {(i0, j0): 1},
                        {(j0, k0+1): 1, (j0+1, k0+1): -1},
                        {(i0, k0+1): 1}
                    )
                    # M4
                    add_term(
                        {(i0+1, j0+1): 1},
                        {(j0+1, k0): 1, (j0, k0): -1},
                        {(i0, k0): 1, (i0+1, k0): 1}
                    )
                    # M5
                    add_term(
                        {(i0, j0): 1, (i0, j0+1): 1},
                        {(j0+1, k0+1): 1},
                        {(i0, k0): -1, (i0, k0+1): 1}
                    )
                    # M6
                    add_term(
                        {(i0+1, j0): 1, (i0, j0): -1},
                        {(j0, k0): 1, (j0, k0+1): 1},
                        {(i0+1, k0+1): 1}
                    )
                    # M7
                    add_term(
                        {(i0, j0+1): 1, (i0+1, j0+1): -1},
                        {(j0+1, k0): 1, (j0+1, k0+1): 1},
                        {(i0, k0): 1}
                    )
                elif ar == 2 and ac == 1 and bc == 2:
                    # (2x1) x (1x2) -> (2x2): 4 terms
                    add_term({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1})
                    add_term({(i0, j0): 1}, {(j0, k0+1): 1}, {(i0, k0+1): 1})
                    add_term({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1})
                    add_term({(i0+1, j0): 1}, {(j0, k0+1): 1}, {(i0+1, k0+1): 1})
                else:
                    # Schoolbook fallback
                    for di in range(ar):
                        for dj in range(ac):
                            for dk in range(bc):
                                add_term(
                                    {(i0+di, j0+dj): 1},
                                    {(j0+dj, k0+dk): 1},
                                    {(i0+di, k0+dk): 1}
                                )
    
    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
# EVOLVE-BLOCK-END
