    # Pruning zero columns
    # Vectorized check: a column is zero if the sum of squares is 0 (for real numbers)
    # Or simply check if any element is non-zero.
    # Since entries are integers, sum of abs is 0 iff all are 0.
    U_nonzero = np.any(U_final != 0, axis=0)
    V_nonzero = np.any(V_final != 0, axis=0)
    W_nonzero = np.any(W_final != 0, axis=0)
    
    keep_mask = U_nonzero & V_nonzero & W_nonzero
    keep = np.where(keep_mask)[0]
    
    U = U_final[:, keep].tolist()
    V = V_final[:, keep].tolist()
    W = W_final[:, keep].tolist()
