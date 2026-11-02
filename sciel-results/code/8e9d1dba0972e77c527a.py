for i in range(n):
    for j in range(m):
        for k in range(p):
            u_row = U_arr[i * m + j]
            v_row = V_arr[j * p + k]
            w_row = W_arr[k * n + i]
            val = np.dot(u_row, v_row * w_row)
            if abs(val - 1.0) > 1e-9:
                return False
