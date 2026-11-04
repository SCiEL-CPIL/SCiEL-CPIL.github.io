import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # 7 terms for 2x2x2 Strassen
        # M1 = (A11 + A22)(B11 + B22)
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M2 = A22(B21 - B11)
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        # M3 = (A12 - A22)B22
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M4 = A11(B12 - B22)
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        # M5 = (A11 + A12)B11
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        # M6 = (A21 - A11)(B11 + B12)
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        # M7 = (A12 - A22)(B21 + B22)
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    # Apply Strassen to (i=0,1; j=0,1; k=0,1)
    add_strassen(0, 1, 0, 1, 0, 1)
    # Apply Strassen to (i=0,1; j=0,1; k=2,3)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Schoolbook for remaining parts involving j=2 or k=4
    # A[:,0:2] * B[0:2, 4:5] -> C[:, 4:5]
    add_schoolbook([0, 1], [0, 1], [4])
    # A[:,0:2] * B[2:3, 0:2] -> C[:, 0:2]
    add_schoolbook([0, 1], [2], [0, 1])
    # A[:,0:2] * B[2:3, 2:4] -> C[:, 2:4]
    add_schoolbook([0, 1], [2], [2, 3])
    # A[:,0:2] * B[2:3, 4:5] -> C[:, 4:5]
    add_schoolbook([0, 1], [2], [4])

    # Verify rank-28
    res = np.zeros((n, m, p))
    for c in range(rank_28):
        Uc = U28[:, c].reshape(n, m)
        Vc = V28[:, c].reshape(m, p)
        Wc = W28[:, c].reshape(p, n)
        res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
    
    # The target tensor is 1s where indices match
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
                
    # Check if res matches target
    # Note: The tensor T has T[i,j,k]=1 for all i,j,k in the sum definition?
    # No, T[i,j,k] = 1 if i,j,k form a valid path.
    # In the sum sum_{i,j,k} A[i,j] B[j,k] W[k,i], the coefficient is 1 for all i,j,k.
    # So the target tensor is indeed all 1s.
    
    if not np.allclose(res, target, atol=1e-9):
        # Fallback to schoolbook rank 30 if something went wrong
        U30 = np.zeros((n * m, 30))
        V30 = np.zeros((m * p, 30))
        W30 = np.zeros((p * n, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j, col] = 1
                    V30[j*p+k, col] = 1
                    W30[k*n+i, col] = 1
                    col += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
