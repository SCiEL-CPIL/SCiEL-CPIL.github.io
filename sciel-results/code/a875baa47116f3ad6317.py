    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U_new[i*m+j][c] * V_new[j*p+k][c] * W_new[k*n+i][c]
                if total != 1:
                    valid = False
                    break
