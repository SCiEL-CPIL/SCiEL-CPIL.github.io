for i in range(n):
    for j in range(m):
        for k in range(p):
            u_val = U_np[i * m + j]
            v_val = V_np[j * p + k]
            w_val = W_np[k * n + i]
            total = np.dot(u_val * v_val, w_val)
            if abs(total - 1.0) > 1e-9:
                return False
