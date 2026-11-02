for j in range(m):
    u_row = U_arr[i * m + j, :]
    v_row = V_arr[j * p + k, :]
    w_val = W_arr[k * n + i, :]
    val += np.sum(u_row * v_row * w_val)
