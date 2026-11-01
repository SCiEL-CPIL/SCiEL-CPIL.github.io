col = 0
for i in range(n):
    for j in range(m):
        for k in range(p):
            U_sb[i*m+j][col] = 1
            V_sb[j*p+k][col] = 1
            W_sb[k*n+i][col] = 1
            col += 1
