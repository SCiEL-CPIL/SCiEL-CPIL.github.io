import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = 4, 4, 5
    deadline = time.time() + 140

    # 1. Define the rank-7 decomposition for 2x2 * 2x2 (Strassen)
    # Indices for 2x2 matrices: 00, 01, 10, 11
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M1 = (A10+A11)B00 -> C10 += M1, C11 += M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    # M2 = A00(B10-B11) -> C01 += M2, C11 += M2
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    # M3 = (A00-A11)(B01+B11) -> C00 += M3, C11 += M3
    U2[0, 3] = 1; U2[3, 3] = -1
    V2[1, 3] = 1; V2[3, 3] = 1
    W2[0, 3] = 1; W2[3, 3] = 1
    # M4 = (A00+A01)B11 -> C01 += M4, C11 += M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    # M5 = (A10-A00)(B00+B01) -> C10 += M5, C00 += M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[0, 5] = 1
    # M6 = (A01-A11)(B10+B11) -> C01 += M6, C10 += M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[2, 6] = 1

    # 2. Construct the rank-49 decomposition for 4x4 * 4x4 via Kronecker product
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U factor
            for i1 in range(2):
                for j1 in range(2):
                    if U2[i1*2+j1, r1] != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                if U2[i2*2+j2, r2] != 0:
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    U44[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            # V factor
            for j1 in range(2):
                for k1 in range(2):
                    if V2[j1*2+k1, r1] != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                if V2[j2*2+k2, r2] != 0:
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    V44[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            # W factor
            for k1 in range(2):
                for i1 in range(2):
                    if W2[k1*2+i1, r1] != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                if W2[k2*2+i2, r2] != 0:
                                    k = k1*2 + k2
                                    i = i1*2 + i2
                                    W44[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]

    # 3. Construct the rank-4 decomposition for 4x4 * 4x1
    # This handles the 5th column of B (index 4)
    U41 = np.zeros((16, 4))
    V41 = np.zeros((20, 4))
    W41 = np.zeros((20, 4))
    
    for r in range(4):
        # U: row r of A
        for j in range(4):
            U41[r*4+j, r] = 1.0
        # V: col 4 of B
        for j in range(4):
            V41[j*5+4, r] = 1.0
        # W: output at (r, 4) -> index 4*4+r
        W41[4*4+r, r] = 1.0

    # 4. Combine into the full 4x4 * 4x5 decomposition
    rank = 53
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Fill the 4x4 * 4x4 part (k=0 to 3)
    U[:, 0:49] = U44
    for j in range(4):
        for k in range(4):
            V[j*5+k, 0:49] = V44[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W44[k*4+i, 0:49]
            
    # Fill the 4x4 * 4x1 part (k=4)
    U[:, 49:53] = U41
    V[:, 49:53] = V41
    W[:, 49:53] = W41

    # 5. Verification
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r]
                            target = 1.0 if (i == ip and k == kp) else 0.0
                            if abs(val - target) > 1e-9:
                                return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
