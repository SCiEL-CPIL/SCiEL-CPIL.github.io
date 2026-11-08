for j_block in range(2):
    for k_block in range(2):
        for t in range(7):
            for l in range(4):
                if strassen_U[t][l] != 0:
                    li = l // 2
                    lj = l % 2
                    gi = li
                    gj = j_block * 2 + lj
                    U[gi * m + gj][col] += strassen_U[t][l]
                
                if strassen_V[t][l] != 0:
                    lj = l // 2
                    lk = l % 2
                    gj = j_block * 2 + lj
                    gk = k_block * 2 + lk
                    V[gj * p + gk][col] += strassen_V[t][l]
                
                if strassen_W[t][l] != 0:
                    li = l // 2
                    lk = l % 2
                    gi = li
                    gk = k_block * 2 + lk
                    W[gk * n + gi][col] += strassen_W[t][l]
            
            col += 1
