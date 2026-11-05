import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    # Use a block-Winograd approach
    # Decompose the 5x5 multiplication using smaller blocks
    # 5 = 3 + 2
    # A is 2x5 = [A_left (2x3) | A_right (2x2)]
    # B is 5x5 = [[B11 (3x3), B12 (3x2)], [B21 (2x3), B22 (2x2)]]
    # C = A*B = [A_left*B11 + A_right*B21, A_left*B12 + A_right*B22]
    
    # This doesn't directly reduce rank.
    
    # Let me try: use the fact that we can share computations.
    # For 2x5x5, schoolbook is 50. 
    # Can we do better? 
    
    # Try a decomposition of rank 48 or so by merging some terms.
    
    # Actually, let me just implement the WFTA literally as described and see what happens.
    # Even if the math isn't quite right, maybe it'll produce a valid decomposition.
    
    N = M = P = 8  # pad to 8x8x8
    
    # Compute the 3D DFT of the tensor
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_hat = np.fft.fftn(T)
    
    # Find non-zero components
    thresh = 1e-10
    nz = np.argwhere(np.abs(T_hat) > thresh)
    rank = len(nz)
    
    if rank >= n * m * p:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Construct the decomposition using DFT
    # T[i,j,k] = (1/(N*M*P)) * sum_{a,b,c} T_hat[a,b,c] * exp(2πi(ia/N + jb/M + kc/P))
    # We want: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # 
    # This doesn't factor cleanly because j appears in both U and V.
    # The DFT approach doesn't directly give a matrix multiplication tensor decomposition.
    
    # So let's just use schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
