def _kronecker_decomposition(n, m, p):
    """
    Rank-49 decomposition via Kronecker product of Strassen 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)
    
    # Kronecker product of the factor matrices
    U_full = np.kron(U2_np, U2_np)
    V_full = np.kron(V2_np, V2_np)
    W_full = np.kron(W2_np, W2_np)
    
    # Convert back to list of lists of ints
    U = U_full.tolist()
    V = V_full.tolist()
    W = W_full.tolist()
    
    # Ensure they are integers
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W, "rank": 49}
