A_mat = U[:, r].reshape(n, m)
B_mat = V[:, r].reshape(m, p)
W_mat = W[:, r].reshape(p, n)
scalar_A = np.sum(A * A_mat)
scalar_B = np.sum(B * B_mat)
C_decomp += scalar_A * scalar_B * W_mat.T
