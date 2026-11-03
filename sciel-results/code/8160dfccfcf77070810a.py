     def add_rank1_2x1x4(i_off, j_off, k_off):
         nonlocal col
         for ii in range(2):
             U[(i_off + ii) * m + j_off][col] = 1
         for kk in range(4):
             V[j_off * p + (k_off + kk)][col] = 1
         for kk in range(4):
             for ii in range(2):
                 W[(k_off + kk) * n + (i_off + ii)][col] = 1
         col += 1
     