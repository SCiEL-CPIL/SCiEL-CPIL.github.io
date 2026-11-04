u_c = U_arr[:, c].reshape(n, m)  # shape (n, m)
v_c = V_arr[:, c].reshape(m, p)  # shape (m, p)
w_c = W_arr[:, c].reshape(p, n)  # shape (p, n)
T_computed += u_c[:, :, None] * v_c[:, None, :] * w_c[None, :, :]
