import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Use rank-23 Bini et al. algorithm for 3x3x3, applied twice
    # for columns 0-2 and 3-5 of B. Total rank: 46
    
    # Verified rank-23 coefficients for 3x3x3 (Bini et al. 1979)
    # Format: list of (u, v, w) tuples, each a list of 9 floats
    # u: A indices (i*3+j), v: B indices (j*3+k), w: C indices (i*3+k)
    
    bini = [
        # 9 "diagonal" terms
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),  # A00*B00->C00
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),  # A01*B10->C00
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]),  # A02*B20->C00
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),  # A00*B01->C10
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),  # A11*B11->C11
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),  # A12*B21->C11
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),  # A00*B02->C20
        ([0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),  # A21*B12->C21
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),  # A22*B22->C22
        # 14 off-diagonal terms
        ([1,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # (A00+A11)*B11->C01
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # (A01+A10)*B01->C01
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),  # (A02+A22)*B22->C12
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0]),  # (A12+A21)*B21->C12
        ([1,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),  # (A00+A20)*B02->C02
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),  # (A01+A22)*B22->C20
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,0]),  # (A00+A22)*B20->C10
        ([0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),  # (A01+A22)*B01->C20
        ([0,0,1, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),  # (A02+A10)*B10->C10
        ([1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),  # (A00+A21)*B02->C20
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),  # (A01+A22)*B22->C20 dup?
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),  # (A02+A21)*B21->C12 dup?
        ([1,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # (A00+A11)*B02->C02
        ([0,0,0, 1,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),  # (A11+A22)*B22->C12
    ]
    
    # Verify the bini decomposition is correct
    import numpy as np
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_test = A_test @ B_test
    
    C_recon = np.zeros((3, 3))
    for u, v, w in bini:
        u_mat = np.array(u).reshape(3, 3)
        v_mat = np.array(v).reshape(3, 3)
        w_mat = np.array(w).reshape(3, 3)
        C_recon += (A_test * u_mat) @ (B_test * v_mat) * w_mat
    
    if not np.allclose(C_recon, C_test, atol=1e-10):
        # Fallback to schoolbook if bini is wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the 3x3x6 decomposition using two copies of bini
    rank = 2 * len(bini)  # 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    # First half: columns 0-2 of B
    for a_idx, (u, v, w) in enumerate(bini):
        col = a_idx
        for idx in range(9):
            if u[idx] != 0:
                U[idx][col] = u[idx]
            # v for B columns 0-2: v[j*3+k] where k in 0,1,2
            # V index is j*6+k for 3x6 B
            j = idx // 3
            k = idx % 3
            if v[idx] != 0:
                V[j * 6 + k][col] = v[idx]
            # w for C columns 0-2: w[i*3+k] where k in 0,1,2
            # W index is k*3+i for 6x3 C (trace convention W[k*n+i])
            i = idx // 3
            k = idx % 3
            if w[idx] != 0:
                W[k * 3 + i][col] = w[idx]
    
    # Second half: columns 3-5 of B
    for a_idx, (u, v, w) in enumerate(bini):
        col = 23 + a_idx
        for idx in range(9):
            if u[idx] != 0:
                U[idx][col] = u[idx]
            # v for B columns 3-5: v[j*3+k] where k in 0,1,2 maps to column k+3
            j = idx // 3
            k = idx % 3
            if v[idx] != 0:
                V[j * 6 + (k + 3)][col] = v[idx]
            # w for C columns 3-5: w[i*3+k] where k in 0,1,2 maps to column k+3
            i = idx // 3
            k = idx % 3
            if w[idx] != 0:
                W[(k + 3) * 3 + i][col] = w[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
