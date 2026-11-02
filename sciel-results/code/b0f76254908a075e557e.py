for i in range(n):
    for j in range(m):
        for k in range(p):
            val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
            if val != 1:
                return False
