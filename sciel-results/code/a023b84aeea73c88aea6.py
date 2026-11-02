U_mat = U.reshape(n, m, rank)
V_mat = V.reshape(m, p, rank)
W_mat = W.reshape(p, n, rank)
T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, V_mat, W_mat)
