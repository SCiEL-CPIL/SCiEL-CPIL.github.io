C_computed = np.zeros((n, p))
for i in range(n):
    for j in range(m):
        for k in range(p):
            C_computed[i, k] += U_reduced[i*m+j, c] * V_reduced[j*p+k, c] * W_reduced[k*n+i, c]
