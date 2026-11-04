for i in range(n):
    for j in range(m):
        for k in range(p):
            val = 0
            for c in range(rank):
                val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
            max_err = max(max_err, abs(val - 1))
