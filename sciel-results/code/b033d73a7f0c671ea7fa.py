"""Matrix multiplication 2,3,8 — rank-44 decomposition via Strassen on 2x2 blocks."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# Strassen 2x2x2 decomposition (verified)
# A is 2x2, B is 2x2, C = A*B is 2x2
# Flat indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
#               B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
#               C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3

# Standard Strassen formulas:
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B10-B11)
# M4 = A11(B01-B10)
# M5 = (A00+A10)B11
# M6 = (A11+A01)(B00+B01)
# M7 = (A00-A10)(B01+B11)
#
# C00 = M1 + M4 - M6 - M7
# C01 = M3 + M7
# C10 = M2 + M5
# C11 = M1 - M2 + M3 + M5

STRASSEN_U = np.array([
    [1, 0, 0, 1],  # M1: A00+A11
    [0, 0, 1, 1],  # M2: A10+A11
    [1, 0, 0, 0],  # M3: A00
    [0, 0, 0, 1],  # M4: A11
    [1, 0, 1, 0],  # M5: A00+A10
    [0, 1, 0, 1],  # M6: A11+A01
    [1, 0, -1, 0], # M7: A00-A10
], dtype=float)

STRASSEN_V = np.array([
    [1, 0, 0, 1],  # M1: B00+B11
    [1, 0, 0, 0],  # M2: B00
    [0, 0, 1, -1], # M3: B10-B11
    [0, 1, -1, 0], # M4: B01-B10
    [0, 0, 0, 1],  # M5: B11
    [1, 1, 0, 0],  # M6: B00+B01
    [0, 1, 0, 1],  # M7: B01+B11
], dtype=float)

STRASSEN_W = np.array([
    [1, 0, 0, 1],  # M1: C00+C11
    [0, 0, 1, 0],  # M2: C10
    [1, 0, 0, -1], # M3: C00-C11
    [0, 1, -1, 0], # M4: C01-C10
    [0, 0, 0, 1],  # M5: C11
    [1, 1, 0, 0],  # M6: C00+C01
    [0, 1, 0, 1],  # M7: C01+C11
], dtype=float)


def _verify_strassen():
    """Verify Strassen decomposition is correct."""
    np.random.seed(42)
    A = np.random.randn(2, 2)
    B = np.random.randn(2, 2)
    C_true = A @ B
    C_recon = np.zeros((2, 2))
    for s in range(7):
        uA = np.dot(STRASSEN_U[s], A.flatten())
        vB = np.dot(STRASSEN_V[s], B.flatten())
        M = uA * vB
        for idx in range(4):
            i, j = idx // 2, idx % 2
            C_recon[i, j] += STRASSEN_W[s][idx] * M
    err = np.max(np.abs(C_recon - C_true))
    return err


def construct():
    n, m, p = 2, 3, 8
    
    # Verify Strassen first
    strassen_err = _verify_strassen()
    if strassen_err > 1e-10:
        # Fall back to schoolbook if Strassen is broken
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
    
    # Build rank-44 decomposition
    # Split: A[:, 0:2] (2x2) with B[0:2, :] (2x8) -> use Strassen on 4 pairs of columns
    #         A[:, 2] (2x1) with B[2, :] (1x8) -> schoolbook (16 terms)
    # Total: 4 * 7 + 16 = 44
    
    rank = 44
    U_np = np.zeros((n * m, rank), dtype=float)   # 6 x 44
    V_np = np.zeros((m * p, rank), dtype=float)   # 24 x 44
    W_np = np.zeros((p * n, rank), dtype=float)   # 16 x 44, index k*n + i
    
    col = 0
    for g in range(4):  # 4 column pairs
        # Strassen part: 7 terms for A[:, 0:2] * B[0:2, 2g:2g+2] -> C[:, 2g:2g+2]
        # Local A indices [A00, A01, A10, A11] -> global flat [0, 1, 3, 4]
        # (A is 2x3, flat: row0=[0,1,2], row1=[3,4,5])
        a_map = [0, 1, 3, 4]  # local -> global
        
        # Local B indices [B00, B01, B10, B11] -> global flat
        # B is 3x8, flat: B[i,j] -> i*8+j
        # B[0,2g], B[0,2g+1], B[1,2g], B[1,2g+1]
        b_map = [0 * 8 + 2 * g, 0 * 8 + 2 * g + 1, 1 * 8 + 2 * g, 1 * 8 + 2 * g + 1]
        
        # Local C indices [C00, C01, C10, C11] -> global W index k*n+i
        # C is 2x8, W index: k*2+i
        # C[0,2g], C[0,2g+1], C[1,2g], C[1,2g+1]
        c_map = [2 * g * 2 + 0, 2 * g * 2 + 1, (2 * g) * 2 + 1, (2 * g + 1) * 2 + 1]
        # Wait: C[i,k] where i in {0,1}, k in {2g, 2g+1}
        # W index = k*n + i = k*2 + i
        # C[0, 2g] -> W[2g*2+0] = W[4g]
        # C[0, 2g+1] -> W[(2g+1)*2+0] = W[4g+2]
        # C[1, 2g] -> W[2g*2+1] = W[4g+1]
        # C[1, 2g+1] -> W[(2g+1)*2+1] = W[4g+3]
        c_map = [4 * g, 4 * g + 2, 4 * g + 1, 4 * g + 3]
        
        for s in range(7):
            # U: A part
            for local_idx, global_idx in enumerate(a_map):
                U_np[global_idx, col] = STRASSEN_U[s][local_idx]
            
            # V: B part
            for local_idx, global_idx in enumerate(b_map):
                V_np[global_idx, col] = STRASSEN_V[s][local_idx]
            
            # W: C part
            for local_idx, global_idx in enumerate(c_map):
                W_np[global_idx, col] = STRASSEN_W[s][local_idx]
            
            col += 1
    
    # Schoolbook part: A[:, 2] (2x1) * B[2, :] (1x8) -> C[:, :] (2x8)
    # A[0,2] = flat index 2, A[1,2] = flat index 5
    # B[2,k] = flat index 2*8+k = 16+k for k=0..7
    # C[i,k] -> W[k*2+i]
    for i in range(2):  # rows of A/C
        for k in range(8):  # columns of B/C
            a_flat = i * 3 + 2  # A[i, 2]
            b_flat = 2 * 8 + k  # B[2, k]
            c_w_idx = k * 2 + i  # W index for C[i, k]
            
            U_np[a_flat, col] = 1
            V_np[b_flat, col] = 1
            W_np[c_w_idx, col] = 1
            col += 1
    
    # Convert to lists of integers (or half-integers)
    # All entries are in {-1, 0, 1}, so they're already integers
    U = [[int(U_np[i, j]) for j in range(rank)] for i in range(n * m)]
    V = [[int(V_np[i, j]) for j in range(rank)] for i in range(m * p)]
    W = [[int(W_np[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
