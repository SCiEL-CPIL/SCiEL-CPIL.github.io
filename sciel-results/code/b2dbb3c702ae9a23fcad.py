import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm on 6 independent 2x2 column-pairs.
    Total rank = 6 * 7 = 42.
    
    The Kronecker product factorization (2x2 matmul rank-7) ⊗ (1x6 scaling rank-6)
    would give rank 42 as well, but the block-diagonal Strassen construction is 
    equivalent and simpler to verify.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    t_start = time.time()
    deadline = t_start + 115  # safety margin before 120s limit
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)   # 4 x 42
    V = np.zeros((m * p, rank), dtype=float)   # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)   # 24 x 42
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # For matrices A (n x m) and B (m x p) where n=m=p=2:
    # C = A * B
    #
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #
    # U indices: (row_in_A, col_in_A) -> flattened as row*m + col
    #   U[0] = (0,0)=a, U[1] = (0,1)=b, U[2] = (1,0)=c, U[3] = (1,1)=d
    # V indices: (row_in_B, col_in_B) -> flattened as row*p + col
    #   V[0] = (0,0)=e, V[1] = (0,1)=f, V[2] = (1,0)=g, V[3] = (1,1)=h
    # W indices: (row_in_C, col_in_C) -> flattened as row*n + col
    #   W[0] = (0,0), W[1] = (0,1), W[2] = (1,0), W[3] = (1,1)
    #
    # Strassen products:
    # M1 = (a+d)(e+h)     -> contributes to all of C
    # M2 = (c+d)e         -> contributes to C[1,0], C[1,1]
    # M3 = a(f-h)         -> contributes to C[0,1], C[1,1]
    # M4 = d(g-e)         -> contributes to C[0,0], C[0,1]
    # M5 = (a+b)h         -> contributes to C[0,0], C[1,0]
    # M6 = (c-a)(e+f)     -> contributes to C[1,0]
    # M7 = (b-d)(g+h)     -> contributes to C[0,1]
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4 + M7
    # C[1,0] = M2 - M5 + M6
    # C[1,1] = M1 - M2 + M3 + M4
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (a+d)(e+h)
                U[0, c] = 1   # a
                U[3, c] = 1   # d
                V[2*t, c] = 1       # e (row 0, col 0 of B block)
                V[p + 2*t+1, c] = 1 # h (row 1, col 1 of B block)
                W[2*t*n + 0, c] = 1    # C[0,0]
                W[2*t*n + 1, c] = -1   # C[0,1]
                W[(2*t+1)*n + 0, c] = -1 # C[1,0]
                W[(2*t+1)*n + 1, c] = 1  # C[1,1]
            elif s == 1:
                # M2 = (c+d)e
                U[2, c] = 1   # c
                U[3, c] = 1   # d
                V[2*t, c] = 1       # e
                W[2*t*n + 1, c] = -1   # C[0,1]: -M2? No, let me re-derive
                # C[1,0] = M2 - M5 + M6, C[1,1] = M1 - M2 + M3 + M4
                # So M2 appears in C[1,0] with +1, C[1,1] with -1
                W[2*t*n + 1, c] = 0
                W[(2*t+1)*n + 0, c] = 1   # C[1,0]
                W[(2*t+1)*n + 1, c] = -1  # C[1,1]
            elif s == 2:
                # M3 = a(f-h)
                U[0, c] = 1   # a
                V[2*t+1, c] = 1       # f (row 0, col 1 of B block)
                V[p + 2*t+1, c] = -1  # h (row 1, col 1 of B block)
                # C[0,1] = M3 + M4 + M7, C[1,1] = M1 - M2 + M3 + M4
                W[2*t*n + 1, c] = 1   # C[0,1]
                W[(2*t+1)*n + 1, c] = 1  # C[1,1]
            elif s == 3:
                # M4 = d(g-e)
                U[3, c] = 1   # d
                V[p + 2*t, c] = 1     # g (row 1, col 0 of B block)
                V[2*t, c] = -1        # e (row 0, col 0 of B block)
                # C[0,0] = M1 - M4 + M5 - M7, C[0,1] = M3 + M4 + M7
                # C[1,1] = M1 - M2 + M3 + M4
                W[2*t*n + 0, c] = -1   # C[0,0]
                W[2*t*n + 1, c] = 1    # C[0,1]
                W[(2*t+1)*n + 1, c] = 1 # C[1,1]
            elif s == 4:
                # M5 = (a+b)h
                U[0, c] = 1   # a
                U[1, c] = 1   # b
                V[p + 2*t+1, c] = 1  # h
                # C[0,0] = M1 - M4 + M5 - M7, C[1,0] = M2 - M5 + M6
                W[2*t*n + 0, c] = 1    # C[0,0]
                W[(2*t+1)*n + 0, c] = -1 # C[1,0]
            elif s == 5:
                # M6 = (c-a)(e+f)
                U[2, c] = 1   # c
                U[0, c] = -1  # a
                V[2*t, c] = 1       # e
                V[2*t+1, c] = 1     # f
                # C[1,0] = M2 - M5 + M6
                W[(2*t+1)*n + 0, c] = 1
            elif s == 6:
                # M7 = (b-d)(g+h)
                U[1, c] = 1   # b
                U[3, c] = -1  # d
                V[p + 2*t, c] = 1     # g
                V[p + 2*t+1, c] = 1   # h
                # C[0,0] = M1 - M4 + M5 - M7, C[0,1] = M3 + M4 + M7
                W[2*t*n + 0, c] = -1   # C[0,0]
                W[2*t*n + 1, c] = 1    # C[0,1]
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fall back to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=float)
            V_s = np.zeros((m * p, rank_school), dtype=float)
            W_s = np.zeros((p * n, rank_school), dtype=float)
            c = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_s[ii*m+jj, c] = 1
                        V_s[jj*p+kk, c] = 1
                        W_s[kk*n+ii, c] = 1
                        c += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V_s.tolist() if 'V_s' in dir() else U.tolist() and U.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
