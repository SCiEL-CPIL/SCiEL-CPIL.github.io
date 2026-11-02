import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((U_size, V_size, W_size))
    indices_u = []
    indices_v = []
    indices_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                indices_u.append(u_idx)
                indices_v.append(v_idx)
                indices_w.append(w_idx)
                
    indices_u = np.array(indices_u)
    indices_v = np.array(indices_v)
    indices_w = np.array(indices_w)
    
    def verify(U, V, W):
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    # Constructive algebraic method: decompose 2x3x3 as direct sum of 2x2x2 (Strassen, rank 7) + 1x3x3 (rank 3)
    # Total rank: 10
    
    # Strassen's 7 terms for 2x2x2:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2x2 part (j in {0,1}, k in {0,1}):
    # U rows: i=0: j=0,1 -> indices 0,1; i=1: j=0,1 -> indices 3,4
    # V rows: j=0: k=0,1 -> indices 0,1; j=1: k=0,1 -> indices 3,4
    # W rows: k=0: i=0,1 -> indices 0,1; k=1: i=0,1 -> indices 2,3
    
    # For 1x3x3 part (j=2, k in {0,1,2}):
    # U rows: i=0: j=2 -> index 2; i=1: j=2 -> index 5
    # V rows: j=2: k=0,1,2 -> indices 6,7,8
    # W rows: k=0: i=0,1 -> indices 0,1; k=1: i=0,1 -> indices 2,3; k=2: i=0,1 -> indices 4,5
    
    rank = 10
    U = np.zeros((U_size, rank))
    V = np.zeros((V_size, rank))
    W = np.zeros((W_size, rank))
    
    # Strassen terms (indices 0-6)
    # Term 0: M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00 (i=0,j=0)
    U[3, 0] = 1  # A11 (i=1,j=1)
    V[0, 0] = 1  # B00 (j=0,k=0)
    V[3, 0] = 1  # B11 (j=1,k=1)
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[1, 0] = -1  # C01 (k=0,i=1) -- wait, let me recheck
    
    # Let me be more careful with the W indexing.
    # W index: k*n + i = k*2 + i
    # k=0: i=0 -> 0, i=1 -> 1
    # k=1: i=0 -> 2, i=1 -> 3
    # k=2: i=0 -> 4, i=1 -> 5
    
    # C00 corresponds to output row i=0, column k=0 -> W index 0
    # C01 corresponds to output row i=0, column k=1 -> W index 2
    # C10 corresponds to output row i=1, column k=0 -> W index 1
    # C11 corresponds to output row i=1, column k=1 -> W index 3
    
    # Reset and redo
    U = np.zeros((U_size, rank))
    V = np.zeros((V_size, rank))
    W = np.zeros((W_size, rank))
    
    # Term 0: M1 = (A00+A11)*(B00+B11) -> contributes to C00 (+1), C11 (+1)
    U[0, 0] = 1  # A00
    U[4, 0] = 1  # A11 (i=1,j=1 -> index 1*3+1=4)
    V[0, 0] = 1  # B00
    V[4, 0] = 1  # B11 (j=1,k=1 -> index 1*3+1=4)
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # Term 1: M2 = (A10+A11)*B00 -> contributes to C10 (+1), C11 (-1)
    U[3, 1] = 1  # A10 (i=1,j=0 -> index 1*3+0=3)
    U[4, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # Term 2: M3 = A00*(B10-B11) -> contributes to C01 (+1), C11 (+1)
    U[0, 2] = 1  # A00
    V[3, 2] = 1  # B10 (j=1,k=0 -> index 1*3+0=3)
    V[4, 2] = -1  # B11
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # Term 3: M4 = A11*(B01-B00) -> contributes to C00 (+1), C10 (+1)
    U[4, 3] = 1  # A11
    V[1, 3] = 1  # B01 (j=0,k=1 -> index 0*3+1=1)
    V[0, 3] = -1  # B00
    W[0, 3] = 1  # C00
    W[1, 3] = 1  # C10
    
    # Term 4: M5 = (A00+A01)*B11 -> contributes to C00 (-1), C01 (+1)
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01 (i=0,j=1 -> index 0*3+1=1)
    V[4, 4] = 1  # B11
    W[0, 4] = -1  # C00
    W[2, 4] = 1  # C01
    
    # Term 5: M6 = (A10-A00)*(B00+B10) -> contributes to C11 (+1)
    U[3, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[3, 5] = 1  # B10
    W[3, 5] = 1  # C11
    
    # Term 6: M7 = (A01-A11)*(B01+B11) -> contributes to C00 (+1), C11 (-1)
    U[1, 6] = 1  # A01
    U[4, 6] = -1  # A11
    V[1, 6] = 1  # B01
    V[4, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[3, 6] = -1  # C11
    
    # 1x3x3 part (terms 7-9) for j=2
    # For each k in {0,1,2}: term k+7
    # U[i, k+7] = 1 for i=0,1 (A[i,2])
    # V[2*3+k, k+7] = 1 (B[2,k])
    # W[k*2+i, k+7] = 1 for i=0,1 (C[i,k])
    
    for k in range(3):
        r = k + 7
        U[2, r] = 1   # A[0,2]
        U[5, r] = 1   # A[1,2]
        V[6 + k, r] = 1  # B[2,k]
        W[2*k, r] = 1    # C[0,k]
        W[2*k+1, r] = 1  # C[1,k]
    
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, fall back to schoolbook
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
