for cbr in range(2):
    for cbc in range(2):
        g = gamma_s[s][cbr][cbc]
        if g == 0: continue
        for li in range(2):
            for lk in range(2):
                k = 2*cbc + lk
                i = 2*cbr + li
                W[k*4 + i, r] = g
