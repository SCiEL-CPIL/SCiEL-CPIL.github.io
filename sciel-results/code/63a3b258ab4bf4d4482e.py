def _kron_combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions via Kronecker product with correct index mapping.
    U1, V1, W1 are for 2x2 matrices.
    U2, V2, W2 are for 2x2 matrices.
    Returns U, V, W for 4x4 matrices.
    """
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 2
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    # Total dimensions
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    
    U_comb = np.zeros((n * m, r), dtype=np.float64)
    V_comb = np.zeros((m * p, r), dtype=np.float64)
    W_comb = np.zeros((p * n, r), dtype=np.float64)
    
    # Indices for 2x2 blocks
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # Full indices
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            
                            # Flattened indices for 4x4
                            idx_U = i * m + j
                            idx_V = j * p + k
                            idx_W = k * n + i
                            
                            # Indices for 2x2 components
                            idx_U1 = i1 * m1 + j1
                            idx_V1 = j1 * p1 + k1
                            idx_W1 = k1 * n1 + i1
                            
                            idx_U2 = i2 * m2 + j2
                            idx_V2 = j2 * p2 + k2
                            idx_W2 = k2 * n2 + i2
                            
                            # Fill in the Kronecker products for all rank components
                            for c1 in range(r1):
                                u1_val = U1[idx_U1, c1]
                                v1_val = V1[idx_V1, c1]
                                w1_val = W1[idx_W1, c1]
                                
                                if u1_val == 0 and v1_val == 0 and w1_val == 0:
                                    continue
                                    
                                for c2 in range(r2):
                                    c = c1 * r2 + c2
                                    u2_val = U2[idx_U2, c2]
                                    v2_val = V2[idx_V2, c2]
                                    w2_val = W2[idx_W2, c2]
                                    
                                    U_comb[idx_U, c] = u1_val * u2_val
                                    V_comb[idx_V, c] = v1_val * v2_val
                                    W_comb[idx_W, c] = w1_val * w2_val
                                    
    return U_comb, V_comb, W_comb
