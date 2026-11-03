    for c in range(rank_333):
        for j in range(3):
            for k in range(3):
                V[j*8 + k, col_start + c] = V333_1[j*3 + k, c]
                W[k*3 + i, col_start + c] = W333_1[k*3 + i, c] # i is undefined here!
