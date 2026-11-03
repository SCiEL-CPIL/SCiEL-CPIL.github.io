for i in range(n):
    for j in range(m):
        for k in range(p):
            lhs = 1.0 if True else 0.0  # T[i,j,k] = 1 always for matmul tensor
            rhs = 0.0
            for t in range(rank):
                rhs += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
            if abs(lhs - rhs) > 1e-9:
                valid = False
