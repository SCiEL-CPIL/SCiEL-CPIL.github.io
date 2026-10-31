M = np.zeros((49, tensor_dim))
for r in range(49):
    u = U_arr[:, r]
    v = V_arr[:, r]
    w = W_arr[:, r]
    outer = (u[:, None, None] * v[None, :, None] * w[None, None, :]).flatten()
    M[r, :] = outer
