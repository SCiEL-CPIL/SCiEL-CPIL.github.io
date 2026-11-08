W_r = W_mat[:, r].reshape(p, n)  # W_r has shape (p, n), so W_r[k, i]
C_computed += np.outer(a_part, b_part) * W_r.T  # W_r.T[i, k] = W_r[k, i]
