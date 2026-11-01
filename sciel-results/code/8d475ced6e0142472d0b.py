def _try_reduce_from_rank39(n, m, p, deadline):
    U39, V39, W39 = _construct_rank39()
    U39_arr = np.array(U39)
    V39_arr = np.array(V39)
    W39_arr = np.array(W39)
    
    # Try removing each column
    for col in range(39):
        if time.time() > deadline:
            break
        U_sub = np.delete(U39_arr, col, axis=1)
        V_sub = np.delete(V39_arr, col, axis=1)
        W_sub = np.delete(W39_arr, col, axis=1)
        
        res = _optimize(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), n, m, p, deadline - 5)
        if res:
            U_opt, V_opt, W_opt = res
            snapped = _snap_and_verify(U_opt, V_opt, W_opt, n, m, p)
            if snapped:
                return snapped
    return None
