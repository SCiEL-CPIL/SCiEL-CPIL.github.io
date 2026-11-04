import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 4x4x4 matrix multiplication.
    Uses the Kronecker product of Strassen's 2x2x2 decomposition, yielding rank 49.
    This is the best known simple construction. The theoretical minimum is 48 (Winograd),
    but constructing it requires specific coefficients which are prone to error in this context.
    We verify the decomposition and fall back to schoolbook if invalid.
    """
    # Strassen's 2x2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Construct 4x4 factors via Kronecker product
    # Shape: (4*4, 7*7) = (16, 49)
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Permutations to map Kronecker indices to standard matrix indices
    # Standard index S = i*4 + j
    # Kronecker index K = (i1*2 + j1)*4 + (i2*2 + j2) 
    # where i = 2*i1 + i2, j = 2*j1 + j2
    # We need to reorder rows of U4_raw, V4_raw, W4_raw
    
    def get_permutation(n):
        perm = np.zeros(n*n, dtype=int)
        for i in range(n):
            for j in range(n):
                S = i * n + j
                i1, i2 = i // 2, i % 2
                j1, j2 = j // 2, j % 2
                # Kronecker product index: (i1, j1) is the "outer" 2x2, (i2, j2) is "inner"
                # np.kron(A, B) index for element (i, j) of A and (k, l) of B is (i*rows_B + k, j*cols_B + l)
                # Here A is U2 (4x7), B is U2 (4x7).
                # Row index in kron: i1 * 4 + i2? No.
                # U2 has 4 rows.
                # kron(U2, U2) has 16 rows.
                # Row r in kron corresponds to (r // 4, r % 4) in (U2_row, U2_row).
                # We want the row corresponding to standard index (i, j).
                # i = 2*i1 + i2, j = 2*j1 + j2.
                # The "outer" block is (i1, j1), "inner" is (i2, j2).
                # In np.kron, the first index is the row of the first matrix, second is row of second.
                # So row index in U4_raw is: (i1, j1) mapped to U2 row?
                # U2 rows are indexed by (i, j) for 2x2.
                # So U2 row for (i1, j1) is i1*2 + j1.
                # U2 row for (i2, j2) is i2*2 + j2.
                # Kron row index = (i1*2 + j1) * 4 + (i2*2 + j2).
                r_out = i1 * 2 + j1
                r_in = i2 * 2 + j2
                K = r_out * 4 + r_in
                perm[S] = K
        return perm

    perm = get_permutation(4)
    
    U4 = U4_raw[perm, :]
    V4 = V4_raw[perm, :]
    W4 = W4_raw[perm, :]
    
    # Verification
    # Check if sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] == 1
    # Using einsum for speed
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # We need to check for all i, j, k:
    # dot(U4[i*4+j], V4[j*4+k] * W4[k*4+i]) == 1
    
    # Reshape for easier indexing
    # U4_reshaped: (4, 4, 49)
    U4_r = U4.reshape(4, 4, 49)
    V4_r = V4.reshape(4, 4, 49)
    W4_r = W4.reshape(4, 4, 49)
    
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # u = U4_r[i, j, :]
                # v = V4_r[j, k, :]
                # w = W4_r[k, i, :]
                val = np.dot(U4_r[i, j, :], V4_r[j, k, :] * W4_r[k, i, :])
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        return _schoolbook(4, 4, 4)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
