def verify(U_mat, V_mat, W_mat, r):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(r):
                    val += U_mat[i*m+j, t] * V_mat[j*p+k, t] * W_mat[k*n+i, t]
                if abs(val - 1) > 1e-9:
                    return False
    return True
