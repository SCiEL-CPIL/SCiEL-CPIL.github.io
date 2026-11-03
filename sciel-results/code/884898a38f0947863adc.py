for br in range(2):
    for bc in range(2):
        for lr in range(2):
            for lc in range(2):
                r = block_idx * 4 + lr * 2 + lc
                i = br * 2 + lr
                j = bc * 2 + lc
                U3[i*4+j][r] = 1
                V3[j][r] = 1
                W3[i][r] = 1
        block_idx += 1
