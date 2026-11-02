<<<<<<< SEARCH
        W_total[b*3, col:col+rank_223] = W_223[0]  # C[0, b*3]
        W_total[b*3+1, col:col+rank_223] = W_223[1] # C[0, b*3+1]
        W_total[b*3+2, col:col+rank_223] = W_223[4] # C[0, b*3+2]
        W_total[9+b*3, col:col+rank_223] = W_223[2] # C[1, b*3]
        W_total[9+b*3+1, col:col+rank_223] = W_223[3] # C[1, b*3+1]
        W_total[9+b*3+2, col:col+rank_223] = W_223[5] # C[1, b*3+2]
=======
        W_total[block*3, col:col+rank_223] = W_223[0]  # C[0, block*3]
        W_total[block*3+1, col:col+rank_223] = W_223[1] # C[0, block*3+1]
        W_total[block*3+2, col:col+rank_223] = W_223[4] # C[0, block*3+2]
        W_total[9+block*3, col:col+rank_223] = W_223[2] # C[1, block*3]
        W_total[9+block*3+1, col:col+rank_223] = W_223[3] # C[1, block*3+1]
        W_total[9+block*3+2, col:col+rank_223] = W_223[5] # C[1, block*3+2]
>>>>>>> REPLACE
