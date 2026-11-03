    sU = [ ... ]
    sV = [ ... ]
    sW = [ ... ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2 # Typo? lj, lk?
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2 # Typo?
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
