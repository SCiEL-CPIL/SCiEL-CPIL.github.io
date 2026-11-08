n, m, p = 2, 4, 4
rank = 28
U = [[0]*rank for _ in range(n*m)]  # 8 x 28
V = [[0]*rank for _ in range(m*p)]  # 16 x 28
W = [[0]*rank for _ in range(p*n)]  # 8 x 28
col = 0
for b in range(2):      # A-column block / B-row block
    for a in range(2):  # output column block
        for r in range(7):
            for (i, j, c) in u_coeffs[r]:
                U[i*m + (2*b + j)][col] = c
            for (j, k, c) in v_coeffs[r]:
                V[(2*b + j)*p + (2*a + k)][col] = c
            for (i, k, c) in w_coeffs[r]:
                W[(2*a + k)*n + i][col] = c
            col += 1
