a_idx_j = a[idx, assigned_facs]
a_j_idx = a[assigned_facs, idx]
p_j = p[assigned_facs]

term1 = np.dot(b[:, p_j], a_idx_j)
