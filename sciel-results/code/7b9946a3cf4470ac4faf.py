# Part 1: 4x4x4 schoolbook (rank 16), columns 0-15
for i in range(4):
    for j in range(4):
        t = i * 4 + j
        U[i*4+j, t] = 1
        for k in range(4):
            V[j*5+k, t] = 1
            W[k*4+i, t] = 1
