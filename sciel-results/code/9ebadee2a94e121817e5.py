rank = 15
U = [[0]*rank for _ in range(15)]  # n*m = 3*5
V = [[0]*rank for _ in range(25)]  # m*p = 5*5
W = [[0]*rank for _ in range(15)]  # p*n = 5*3

for i in range(3):
    for j in range(5):
        c = i*5 + j
        U[i*5 + j][c] = 1
        for l in range(5):
            for k in range(5):
                if l == j:
                    V[l*5 + k][c] = 1
        for k in range(5):
            W[k*3 + i][c] = 1
