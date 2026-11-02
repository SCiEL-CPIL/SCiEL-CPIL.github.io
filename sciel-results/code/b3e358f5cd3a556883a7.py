def add_term(U, V, W, u_idx, v_idx, w_idx, col, val=1):
    U[u_idx][col] += val
    V[v_idx][col] += val
    W[w_idx][col] += val
