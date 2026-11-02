def construct():
    import numpy as np

    n, m, p = 4, 4, 5
    rank = 50
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen terms
    # u, v, w are length 4 vectors
    # Indexing: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Define block-level coefficients for 4x4 Strassen
    # s=0..6
    # alpha: dict block -> coeff
    # beta: dict block -> coeff
    # gamma: dict block -> coeff
    # Blocks: 00, 01, 10, 11
    block_coeffs = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'10': 1, '00': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'01': 1, '00': -1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1})
    ]
    
    # Block index mapping for 4x4 matrices (16 entries)
    # Block 00: indices 0, 1, 4, 5
    # Block 01: indices 2, 3, 6, 7
    # Block 10: indices 8, 9, 12, 13
    # Block 11: indices 10, 11, 14, 15
    block_indices = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    col = 0
    for s in range(7):
        alpha, beta, gamma = block_coeffs[s]
        for t in range(7):
            u2, v2, w2 = strassen_2x2[t]
            
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            # Construct u
            for block, coeff in alpha.items():
                if coeff != 0:
                    for idx, val in zip(block_indices[block], u2):
                        u[idx] += coeff * val
            
            # Construct v
            for block, coeff in beta.items():
                if coeff != 0:
                    for idx, val in zip(block_indices[block], v2):
                        v[idx] += coeff * val
            
            # Construct w
            for block, coeff in gamma.items():
                if coeff != 0:
                    for idx, val in zip(block_indices[block], w2):
                        w[idx] += coeff * val
            
            # Add to U, V, W
            for i in range(16):
                U[i][col] = u[i]
            for i in range(16):
                V[i][col] = v[i]
            for i in range(16):
                W[i][col] = w[i]
            
            col += 1
    
    # Add the extra term for k=4
    # u_extra: all ones length 16
    # v_extra: length 20, 1s at 4, 9, 14, 19
    # w_extra: length 20, 1s at 16, 17, 18, 19
    
    for i in range(16):
        U[i][col] = 1
    
    v_extra = [0] * 20
    for j in range(4):
        v_extra[j * 5 + 4] = 1
    for i in range(20):
        V[i][col] = v_extra[i]
    
    w_extra = [0] * 20
    for i in range(4):
        w_extra[16 + i] = 1
    for i in range(20):
        W[i][col] = w_extra[i]
    
    col += 1
    
    # Verify the decomposition
    # Build target tensor T
    T = np.zeros((16, 20, 20))
    for i in range(4):
        for j in range(4):
            for k in range(5):
                a = i * 4 + j
                b = j * 5 + k
                c = k * 4 + i
                T[a, b, c] = 1
    
    # Reconstruct T from U, V, W
    T_recon = np.zeros((16, 20, 20))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for c_idx in range(rank):
        # Outer product of three vectors
        term = np.einsum('a,b,c->abc', U_arr[:, c_idx], V_arr[:, c_idx], W_arr[:, c_idx])
        T_recon += term
    
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to integer/half-integer lists
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
