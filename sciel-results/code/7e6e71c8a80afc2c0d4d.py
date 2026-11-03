R = n * m * p # 2*5*6 = 60
# U_fb shape (10, 60)
# V_fb shape (30, 60)
# W_fb shape (12, 60)
# Logic:
for i in range(n):
    for j in range(m):
        for k in range(p):
            U_fb[i * m + j][idx] = 1.0
            V_fb[j * p + k][idx] = 1.0
            W_fb[k * n + i][idx] = 1.0
            idx += 1
