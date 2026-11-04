<<<<<<< SEARCH
    # Construct 4x4 decomposition using Kronecker product structure
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            rU = i * 4 + j
                            U4[rU, c] += U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = 2 * k1 + k2
                                    rV = j * 4 + k
                                    V4[rV, c] += V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                                    
                                    rW = k * 4 + i
                                    W4[rW, c] += W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
=======
    # Construct 4x4 decomposition using Kronecker product structure
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permute rows to match row-major indexing of 4x4 matrices
    # U: A (4x4). Kron index (b, e). b=2*i_b+j_b, e=2*i_e+j_e.
    # Actual index: i_full=2*i_b+i_e, j_full=2*j_b+j_e. Row = 4*i_full + j_full.
    map_U = np.zeros(16, dtype=int)
    for i_b in range(2):
        for j_b in range(2):
            b = 2 * i_b + j_b
            for i_e in range(2):
                for j_e in range(2):
                    e = 2 * i_e + j_e
                    r_kron = b * 4 + e
                    r_actual = (2 * i_b + i_e) * 4 + (2 * j_b + j_e)
                    map_U[r_kron] = r_actual
    
    U4 = np.zeros_like(U4_kron)
    for r_k in range(16):
        U4[map_U[r_k]] = U4_kron[r_k]
        
    # V: B (4x4). Kron index (b_v, e_v). b_v=2*j_b+k_b, e_v=2*j_e+k_e.
    # Actual index: j_full=2*j_b+j_e, k_full=2*k_b+k_e. Row = 4*j_full + k_full.
    map_V = np.zeros(16, dtype=int)
    for j_b in range(2):
        for k_b in range(2):
            b_v = 2 * j_b + k_b
            for j_e in range(2):
                for k_e in range(2):
                    e_v = 2 * j_e + k_e
                    r_kron = b_v * 4 + e_v
                    r_actual = (2 * j_b + j_e) * 4 + (2 * k_b + k_e)
                    map_V[r_kron] = r_actual
    
    V4 = np.zeros_like(V4_kron)
    for r_k in range(16):
        V4[map_V[r_k]] = V4_kron[r_k]
        
    # W: C (4x4). Kron index (b_w, e_w). b_w=2*k_b+i_b, e_w=2*k_e+i_e.
    # Actual index: k_full=2*k_b+k_e, i_full=2*i_b+i_e. Row = 4*k_full + i_full.
    map_W = np.zeros(16, dtype=int)
    for k_b in range(2):
        for i_b in range(2):
            b_w = 2 * k_b + i_b
            for k_e in range(2):
                for i_e in range(2):
                    e_w = 2 * k_e + i_e
                    r_kron = b_w * 4 + e_w
                    r_actual = (2 * k_b + k_e) * 4 + (2 * i_b + i_e)
                    map_W[r_kron] = r_actual
    
    W4 = np.zeros_like(W4_kron)
    for r_k in range(16):
        W4[map_W[r_k]] = W4_kron[r_k]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
>>>>>>> REPLACE
