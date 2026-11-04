def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Vectorized Kronecker construction
    # U2 is (4, 7). Reshape to (2, 2, 7) for (i1, j1, r1)?
    # Actually, U2[i*2+j, r]. 
    # We want U_new[i1*4+i2*2+j1*4+j2 ... wait.
    # Index i = 2*i1 + i2. j = 2*j1 + j2.
    # U_new[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    # Let's just use the loop but optimize it slightly or keep it as is.
    # The loop is 49 * 16 * 16 * 16? No.
    # The loop in current code:
    # for i1... for i2... for j1... for j2... for k1... for k2... for r1... for r2...
    # This is 2*2*2*2*2*2*7*7 = 4096 iterations. Very fast.
    
    # However, we can fill columns r1*7+r2 directly.
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U col r is U2[:, r1] kron U2[:, r2]
            U[:, r] = np.kron(U2[:, r1], U2[:, r2])
            V[:, r] = np.kron(V2[:, r1], V2[:, r2])
            W[:, r] = np.kron(W2[:, r1], W2[:, r2])
            
            if time.time() > deadline:
                return _trivial_decomp(n, m, p)
                
    # Verification
    # T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    # T_true = np.zeros((4, 4, 4))
    # for i in range(4):
    #   for j in range(4):
    #     for k in range(4):
    #       T_true[i, j, k] = 1.0 if False else 0.0 # Matrix mult tensor T[i,j,k] = 1 if ...?
    # The tensor for matrix mult is T[i, j, k] where C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # In the decomposition T = sum_r U_r V_r W_r.
    # U has index (i, j). V has index (j, k). W has index (k, i).
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    # We want T[i, j, k] = 1 if we are calculating C[i, k] from A[i, j] and B[j, k]?
    # No, the tensor is defined such that C = A x B is recovered.
    # C[i, k] = sum_j T[i, j, k] A[i, j] B[j, k].
    # So T[i, j, k] should be 1.
    
    # Let's check indices in W.
    # W[k*n + i, r]. Index is k*4 + i.
    # So W[k, i, r].
    # The sum is sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    # This matches C[i, k] = sum_j A[i, j] B[j, k] if T[i, j, k] = 1.
    
    # Quick verification
    T_decomp = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    T_true = np.ones((4, 4, 4))
    
    if not np.allclose(T_decomp, T_true):
        return _trivial_decomp(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
