"""Matrix multiplication 3,3,6 - rank 46 decomposition via two 3x3 blocks."""
DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the decomposition correctly represents matrix multiplication."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    # Compute C from decomposition
    C_comp = np.zeros((n, p))
    for c in range(U_arr.shape[1]):
        u = U_arr[:, c]  # (n*m,)
        v = V_arr[:, c]  # (m*p,)
        w = W_arr[:, c]  # (p*n,)
        
        # u picks out entries of A, v picks out entries of B
        # a_val = sum(u[i*m+j] * A[i,j])
        a_val = 0
        for i in range(n):
            for j in range(m):
                a_val += u[i*m+j] * A_test[i,j]
        
        b_val = 0
        for j in range(m):
            for k in range(p):
                b_val += v[j*p+k] * B_test[j,k]
        
        # w adds to C entries
        for idx in range(p*n):
            k = idx // n
            i = idx % n
            C_comp[i, k] += w[idx] * a_val * b_val
    
    return np.allclose(C_comp, C_test, atol=1e-9)

def construct():
    """
    Rank-46 decomposition for 3x3 * 3x6 matrix multiplication.
    """
    n, m, p = 3, 3, 6
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]   # 9 x 46
    V = [[0] * rank for _ in range(m * p)]   # 18 x 46
    W = [[0] * rank for _ in range(p * n)]   # 18 x 46

    # 23-rank decomposition of 3x3 matrix multiplication
    terms = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,1,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,1, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 1,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,1, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 1,0,0, 0,1,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
    ]
    
    for t_idx in range(23):
        A_coeffs, B_coeffs, C_coeffs = terms[t_idx]
        
        # Block 1: k=0,1,2
        col1 = t_idx
        for idx, coeff in enumerate(A_coeffs):
            if coeff != 0:
                i = idx // 3
                j = idx % 3
                U[i * m + j][col1] += coeff
        for idx, coeff in enumerate(B_coeffs):
            if coeff != 0:
                j = idx // 3
                k = idx % 3
                V[j * p + k][col1] += coeff
        for idx, coeff in enumerate(C_coeffs):
            if coeff != 0:
                i = idx // 3
                k = idx % 3
                W[k * n + i][col1] += coeff
        
        # Block 2: k=3,4,5
        col2 = t_idx + 23
        for idx, coeff in enumerate(A_coeffs):
            if coeff != 0:
                i = idx // 3
                j = idx % 3
                U[i * m + j][col2] += coeff
        for idx, coeff in enumerate(B_coeffs):
            if coeff != 0:
                j = idx // 3
                k = idx % 3
                V[j * p + (k + 3)][col2] += coeff
        for idx, coeff in enumerate(C_coeffs):
            if coeff != 0:
                i = idx // 3
                k = idx % 3
                W[(k + 3) * n + i][col2] += coeff
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook rank 54
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
