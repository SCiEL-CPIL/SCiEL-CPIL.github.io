import numpy as np
import time

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Split middle dimension m=3 into j=0 and j in {1,2}.
    - Block 1 (j=0): 2x1x5 schoolbook, rank 10.
    - Block 2 (j in {1,2}): 2x2x5.
      - Split k into {0,1}, {2}, {3,4}.
      - Sub-block 2a (k=0,1): 2x2x2 Strassen, rank 7.
      - Sub-block 2b (k=2): 2x2x1 schoolbook, rank 4.
      - Sub-block 2c (k=3,4): 2x2x2 Strassen, rank 7.
    - Total rank: 10 + 7 + 4 + 7 = 28.
    
    Indexing conventions:
    - A is 2x3, flattened row-major: A[i,j] -> index i*3+j
    - B is 3x5, flattened row-major: B[j,k] -> index j*5+k
    - C is 2x5, output follows W[k*n+i, r] convention: C[i,k] -> index k*2+i
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Helper to add a 2x2x2 Strassen block
    # Local A indices: rows 0,1; cols j0, j1
    # Local B indices: rows j0, j1; cols k0, k1
    # Local C indices: rows 0,1; cols k0, k1
    def add_strassen(j0, j1, k0, k1):
        nonlocal col
        
        # M1: (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
        # Contributes to C[0,k0] and C[1,k1]
        U[0 * m + j0][col] = 1
        U[1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2: (A[1,j0] + A[1,j1]) * B[j0,k0]
        # Contributes to C[1,k0] and C[1,k1]
        U[1 * m + j0][col] = 1
        U[1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3: A[0,j0] * (B[j0,k1] - B[j1,k1])
        # Contributes to C[0,k1] and C[1,k1]
        U[0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4: A[1,j1] * (B[j1,k0] - B[j0,k0])
        # Contributes to C[0,k0] and C[1,k0]
        U[1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5: (A[0,j0] + A[0,j1]) * B[j1,k1]
        # Contributes to C[0,k0] and C[0,k1]
        U[0 * m + j0][col] = 1
        U[0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6: (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
        # Contributes to C[1,k1]
        U[1 * m + j0][col] = 1
        U[0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7: (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
        # Contributes to C[0,k0]
        U[0 * m + j1][col] = 1
        U[1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    # Block 2a: j in {1,2}, k in {0,1}
    add_strassen(1, 2, 0, 1)
    
    # Block 2b: j in {1,2}, k=2 — schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    # Block 2c: j in {1,2}, k in {3,4}
    add_strassen(1, 2, 3, 4)
    
    # Verification
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with a few random integer matrices
    verified = True
    for _ in range(3):
        if time.time() > deadline:
            break
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        C_calc = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_vec = U_np[:, r].reshape(n, m)
            v_vec = V_np[:, r].reshape(m, p)
            # W index is k*n + i. So W_np[:, r].reshape(p, n) has element [k, i] = W[k*n+i, r].
            # C[i, k] += S_r * W[k*n+i, r], which is C += S_r * W_mat.T where W_mat[k,i] = W[k*n+i,r]
            s_val = np.dot(u_vec.flatten(), A.flatten()) * np.dot(v_vec.flatten(), B.flatten())
            C_calc += s_val * W_np[:, r].reshape(p, n).T
            
        if not np.allclose(C_calc, C_expected):
            verified = False
            break

    return {"U": U, "V": V, "W": W}
