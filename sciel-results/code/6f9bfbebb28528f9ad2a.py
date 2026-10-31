    # Pre-compute Kronecker products
    U_cols = []
    V_cols = []
    W_cols = []
    
    for s in range(7):
        for t in range(7):
            U_cols.append(np.kron(a_coeffs[s], a_coeffs[t]).flatten())
            V_cols.append(np.kron(b_coeffs[s], b_coeffs[t]).flatten())
            W_cols.append(np.kron(c_coeffs[s], c_coeffs[t]).flatten())
            
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
