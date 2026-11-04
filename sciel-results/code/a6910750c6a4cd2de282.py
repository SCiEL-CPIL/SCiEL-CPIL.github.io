def verify(U, V, W, r):
    # ...
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    # ...
    u_vals = Ua[u_idx, :]
    v_vals = Va[v_idx, :]
    w_vals = Wa[w_idx, :]
    prod = u_vals * v_vals * w_vals
    result = prod.sum(axis=1)
    return np.max(np.abs(result - 1.0))
