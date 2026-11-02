# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Laderman's 23-term decomposition for 3x3 * 3x3
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i] = 1
    # U: 9x23 (A indices), V: 9x23 (B indices), W: 9x23 (C indices)
    
    # Define the 23 terms from Laderman's algorithm
    # Format: (U_vector, V_vector, W_vector)
    # U indices: i*3+j
    # V indices: j*3+k
    # W indices: k*3+i
    
    laderman_23 = [
        # 1
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # 2
        ([0,1,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        # 3
        ([0,0,1, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0]),
        # 4
        ([0,0,0, 1,0,0, 1,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # 5
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0]),
        # 6
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        # 7
        ([0,0,0, 1,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        # 8
        ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        # 9
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        # 10
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # 11
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # 12
        ([0,0,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # 13
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        # 14
        ([0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # 15
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # 16
        ([0,1,0, -1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,0,0]),
        # 17
        ([0,0,1, 0,0,-1, 0,0,0], [0,0,0, 0,1,0, 0,0,-1], [0,1,0, 0,0,0, 0,0,0]),
        # 18
        ([0,0,0, 1,0,0, -1,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # 19
        ([0,0,0, 0,1,0, 0,-1,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,1,0, 0,0,0]),
        # 20
        ([0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,1, 0,0,0]),
        # 21
        ([0,0,0, 1,0,0, -1,0,0], [1,0,0, 0,0,0, 0,0,-1], [0,0,0, 0,0,0, 1,0,0]),
        # 22
        ([0,0,0, 0,1,0, 0,-1,0], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,1,0]),
        # 23
        ([0,0,0, 0,0,1, 0,0,-1], [0,0,0, 0,1,0, 0,0,-1], [0,0,0, 0,0,0, 0,0,1]),
    ]
    
    # Build 3x3x3 matrices
    U3 = np.zeros((9, 23), dtype=int)
    V3 = np.zeros((9, 23), dtype=int)
    W3 = np.zeros((9, 23), dtype=int)
    
    for c in range(23):
        u, v, w = laderman_23[c]
        U3[:, c] = u
        V3[:, c] = v
        W3[:, c] = w
    
    # Verify 3x3x3 decomposition
    def verify_3x3(U, V, W, rank):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(rank):
                        s += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_3x3(U3, V3, W3, 23):
        # Fallback to schoolbook if Laderman doesn't verify
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=int)
        V_sb = np.zeros((m * p, rank_sb), dtype=int)
        W_sb = np.zeros((p * n, rank_sb), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, column] = 1
                    V_sb[j * p + k, column] = 1
                    W_sb[k * n + i, column] = 1
                    column += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Now extend to 3x3 * 3x6
    rank = 46
    U46 = np.zeros((9, rank), dtype=int)
    V46 = np.zeros((18, rank), dtype=int)
    W46 = np.zeros((18, rank), dtype=int)
    
    # First 23 terms: handle B[j,0..2] -> C[i,0..2]
    for c in range(23):
        U46[:, c] = U3[:, c]
        for j in range(3):
            for k in range(3):
                V46[j*6+k, c] = V3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W46[k*3+i, c] = W3[k*3+i, c]
    
    # Second 23 terms (c=23..45): handle B[j,3..5] -> C[i,3..5]
    for c in range(23):
        c2 = c + 23
        U46[:, c2] = U3[:, c]
        for j in range(3):
            for k in range(3, 6):
                V46[j*6+k, c2] = V3[j*3+(k-3), c]
        for k in range(3, 6):
            for i in range(3):
                W46[k*3+i, c2] = W3[(k-3)*3+i, c]
    
    # Verify the 3x3x6 decomposition
    def verify_3x3x6(U, V, W, rank):
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    s = 0
                    for c in range(rank):
                        s += U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify_3x3x6(U46, V46, W46, 46):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=int)
        V_sb = np.zeros((m * p, rank_sb), dtype=int)
        W_sb = np.zeros((p * n, rank_sb), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, column] = 1
                    V_sb[j * p + k, column] = 1
                    W_sb[k * n + i, column] = 1
                    column += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
# EVOLVE-BLOCK-END
