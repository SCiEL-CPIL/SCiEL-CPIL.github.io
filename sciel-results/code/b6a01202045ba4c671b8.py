import numpy as np

def construct():
    n, m, p = 3, 5, 7
    # Pad to 4, 8, 8
    N, M, P = 4, 8, 8
    
    U, V, W = decompose(N, M, P)
    
    # Crop back to 3, 5, 7
    # U shape (N*M, R) -> (n*m, R)
    # Indices: i*M + j. We need i < n, j < m.
    U_crop = np.zeros((n*m, U.shape[1]))
    for i in range(n):
        for j in range(m):
            U_crop[i*m + j] = U[i*M + j]
            
    V_crop = np.zeros((m*p, V.shape[1]))
    for j in range(m):
        for k in range(p):
            V_crop[j*p + k] = V[j*P + k]
            
    W_crop = np.zeros((p*n, W.shape[1]))
    for k in range(p):
        for i in range(n):
            W_crop[k*n + i] = W[k*N + i]
            
    return {"U": U_crop.tolist(), "V": V_crop.tolist(), "W": W_crop.tolist()}

def decompose(n, m, p):
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W
        
    # Split n
    n1 = n // 2
    n2 = n - n1
    # Split m
    m1 = m // 2
    m2 = m - m1
    # Split p
    p1 = p // 2
    p2 = p - p1
    
    # 8 sub-problems? No, just splitting.
    # C = A B
    # A is n x m, B is m x p.
    # Split A into A1 (n1 x m), A2 (n2 x m).
    # C1 = A1 B, C2 = A2 B.
    
    U1, V1, W1 = decompose(n1, m, p)
    U2, V2, W2 = decompose(n2, m, p)
    
    # Concatenate
    # U1 has shape (n1*m, R1), U2 has shape (n2*m, R2)
    # We need to map indices.
    
    # This simple splitting gives rank R1 + R2.
    # R(n,m,p) <= R(n1,m,p) + R(n2,m,p)
    # R(4,8,8) <= R(2,8,8) + R(2,8,8) = 2 R(2,8,8)
    # R(2,8,8) <= R(1,8,8) + R(1,8,8) = 2 R(1,8,8)
    # R(1,8,8) <= 8 R(1,1,8)? No.
    # R(1,8,8) is rank of dot product of length 8 vector and 8x8 matrix?
    # C_k = sum_j A_j B_jk. Rank 8.
    # So R(1,8,8) = 8.
    # R(2,8,8) = 16.
    # R(4,8,8) = 32.
    
    U = np.hstack([U1, U2])
    V = np.vstack([V1, V2]) # No, V is shared?
    # In the decomposition C = A B, the B is the same.
    # So V1 and V2 should be the same?
    # No, the decomposition of A1 B and A2 B might use different V.
    # But we can just concatenate the terms.
    
    # U1 indices: i < n1. Map to global i.
    # U2 indices: i >= n1. Map to global i.
    
    U_full = np.zeros((n*m, U1.shape[1] + U2.shape[1]))
    U_full[:n1*m, :U1.shape[1]] = U1
    U_full[n1*m:, U1.shape[1]:] = U2
    
    V_full = np.vstack([V1, V2]) # This is wrong. V has shape (m*p, R).
    # V1 is for A1 B. V2 is for A2 B.
    # They are separate terms. So we concatenate V1 and V2 horizontally?
    # No, V1 has shape (m*p, R1). V2 has shape (m*p, R2).
    # We want V_full to have shape (m*p, R1+R2).
    V_full = np.hstack([V1, V2])
    
    W1_full = np.zeros((p*n, W1.shape[1]))
    W1_full[:, :n1] = W1 # W1 has shape (p*n1, R1). We need to embed in (p*n, R1).
    # W1 indices: k*n1 + i. Global: k*n + i.
    for k in range(p):
        for i in range(n1):
            W1_full[k*n + i] = W1[k*n1 + i]
            
    W2_full = np.zeros((p*n, W2.shape[1]))
    for k in range(p):
        for i in range(n2):
            W2_full[k*n + (n1+i)] = W2[k*n2 + i]
            
    W_full = np.hstack([W1_full, W2_full])
    
    return U_full, V_full, W_full
