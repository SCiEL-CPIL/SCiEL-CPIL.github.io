def construct():
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 135

    # 1. Get Strassen 2x2x2 matrices
    U2, V2, W2 = strassen_2x2()
    
    # 2. Construct 4x4x4 Strassen matrices
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for l in range(7):
        uA = U2[:, l]
        uB = V2[:, l]
        wC = W2[:, l]
        
        for s in range(7):
            r = l * 7 + s
            uSub = U2[:, s]
            vSub = V2[:, s]
            wSub = W2[:, s]
            
            for i in range(4):
                bi = i // 2
                si = i % 2
                for j in range(4):
                    bj = j // 2
                    sj = j % 2
                    # U index: i * 4 + j (for 4x4 case)
                    # But in the final 4x5 case, U index is i * 5 + j
                    # We are building for 4x4x4, so let's keep local indices
                    # and map later.
                    # Local U index: i * 4 + j
                    idx_U = i * 4 + j
                    U44[idx_U, r] = uA[bi * 2 + bj] * uSub[si * 2 + sj]
                    
            for j in range(4):
                bj = j // 2
                sj = j % 2
                for k in range(4):
                    bk = k // 2
                    sk = k % 2
                    idx_V = j * 4 + k
                    V44[idx_V, r] = uB[bj * 2 + bk] * vSub[sj * 2 + sk]
                    
            for k in range(4):
                bk = k // 2
                sk = k % 2
                for i in range(4):
                    bi = i // 2
                    si = i % 2
                    idx_W = k * 4 + i
                    W44[idx_W, r] = wC[bk * 2 + bi] * wSub[sk * 2 + si]

    # 3. Construct the other parts
    # Part 2: A[:, 0:4] B[0:4, 4] -> C[:, 4]
    # Rank 16
    U_part2 = np.zeros((20, 16))
    V_part2 = np.zeros((25, 16))
    W_part2 = np.zeros((20, 16))
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_part2[i * 5 + j, r] = 1
            V_part2[j * 5 + 4, r] = 1
            W_part2[4 * 4 + i, r] = 1
            
    # Part 3: A[:, 4] B[4, 0:4] -> C[:, 0:4]
    # Rank 16
    U_part3 = np.zeros((20, 16))
    V_part3 = np.zeros((25, 16))
    W_part3 = np.zeros((20, 16))
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U_part3[i * 5 + 4, r] = 1
            V_part3[4 * 5 + k, r] = 1
            W_part3[k * 4 + i, r] = 1
            
    # Part 4: A[:, 4] B[4, 4] -> C[:, 4]
    # Rank 4
    U_part4 = np.zeros((20, 4))
    V_part4 = np.zeros((25, 4))
    W_part4 = np.zeros((20, 4))
    for i in range(4):
        r = i
        U_part4[i * 5 + 4, r] = 1
        V_part4[4 * 5 + 4, r] = 1
        W_part4[4 * 4 + i, r] = 1
        
    # 4. Map U44, V44, W44 to the 4x5x5 indices
    U4 = np.zeros((20, 49))
    V4 = np.zeros((25, 49))
    W4 = np.zeros((20, 49))
    
    for i in range(4):
        for j in range(4):
            U4[i * 5 + j, :] = U44[i * 4 + j, :]
            
    for j in range(4):
        for k in range(4):
            V4[j * 5 + k, :] = V44[j * 4 + k, :]
            
    for k in range(4):
        for i in range(4):
            W4[k * 4 + i, :] = W44[k * 4 + i, :]
            
    # 5. Concatenate
    U = np.hstack([U4, U_part2, U_part3, U_part4])
    V = np.hstack([V4, V_part2, V_part3, V_part4])
    W = np.hstack([W4, W_part2, W_part3, W_part4])
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
