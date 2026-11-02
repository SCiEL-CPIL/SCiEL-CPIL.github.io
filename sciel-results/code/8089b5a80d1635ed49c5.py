    C_computed = np.zeros((n, p))
    for c in range(rank):
        C_computed += W_np[:, c].reshape(p, n).T * (a_forms[c] * b_forms[c])
