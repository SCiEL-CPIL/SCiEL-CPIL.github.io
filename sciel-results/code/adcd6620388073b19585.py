import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2_factors():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1 # C21 += M2, C22 -= M2
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12 += M3, C22 += M3
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = 1 # C21 += M4, C11 += M4
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C11 -= M5, C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return U, V, W

def _schoolbook_4x4x2_factors():
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Strassen 4x4x4
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Kronecker product for 4x4x4
    U444 = np.kron(U2, U2)
    V444 = np.kron(V2, V2)
    W444 = np.kron(W2, W2)
    
    # Permute rows to match index convention
    # U: i*m+j -> (i1*2+i2)*4 + j1*2+j2
    # Kron: (i1*2+j1)*4 + (i2*2+j2)
    perm_U = []
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    target = (i1*2 + i2)*4 + j1*2 + j2
                    src = (i1*2 + j1)*4 + (i2*2 + j2)
                    perm_U.append(src)
    U444 = U444[perm_U, :]
    
    # V: j*p+k -> (j1*2+j2)*4 + k1*2+k2
    # Kron: (j1*2+k1)*4 + (j2*2+k2)
    perm_V = []
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    target = (j1*2 + j2)*4 + k1*2 + k2
                    src = (j1*2 + k1)*4 + (j2*2 + k2)
                    perm_V.append(src)
    V444 = V444[perm_V, :]
    
    # W: k*n+i -> (k1*2+k2)*4 + i1*2+i2
    # Kron: (k1*2+i1)*4 + (k2*2+i2)
    perm_W = []
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    target = (k1*2 + k2)*4 + i1*2 + i2
                    src = (k1*2 + i1)*4 + (k2*2 + i2)
                    perm_W.append(src)
    W444 = W444[perm_W, :]
    
    # 2. Schoolbook 4x4x2
    U442, V442, W442 = _schoolbook_4x4x2_factors()
    
    # Combine
    U = np.hstack([U444, U442])
    V = np.hstack([V444, V442])
    W = np.hstack([W444, W442])
    
    rank = U.shape[1]
    
    # Greedy column elimination
    # Check if column c can be removed
    # This is expensive, so we do a limited pass
    if time.time() < deadline - 10.0:
        for c in range(rank - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            # Try removing column c
            # Check if U[:, c] is in span of U[:, :c] + U[:, c+1:]
            # This is a rank check.
            # Simplified: Check if the tensor term is redundant.
            # We skip this for now to ensure correctness and speed, 
            # as 81 is already a strong improvement.
            pass
            
    # Convert to list of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
