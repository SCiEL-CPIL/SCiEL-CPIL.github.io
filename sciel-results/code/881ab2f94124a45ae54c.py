for i in range(6):
    for j in range(6):
        for k in range(6):
            U[i*6+j, col] = 1
            V[j*6+k, col] = 1
            W[k*6+i, col] = 1
            col += 1
