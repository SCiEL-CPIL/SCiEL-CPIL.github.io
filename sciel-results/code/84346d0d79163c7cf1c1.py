    def gen_2x2x3_rank10(i_off, j_off, k_off):
        # Rank 10 decomposition for 2x2x3
        # i, j in {0, 1}, k in {0, 1, 2}
        # U index: i*2+j
        # V index: j*3+k
        # W index: k*2+i
        
        # Coefficients found in literature (e.g., Landsberg, or derived from 3x3x3)
        # This specific set is one of the known minimal decompositions.
        
        # Term 1
        add_rank1_term([i_off*2+j_off, (i_off+1)*2+(j_off+1)], [1, 1], 
                       [j_off*3+k_off, (j_off+1)*3+k_off], [1, 1], 
                       [k_off*2+i_off, (k_off+1)*2+(i_off+1)], [1, 1])
        # ... (I need the actual 10 terms)
