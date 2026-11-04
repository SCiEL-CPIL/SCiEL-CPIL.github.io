import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 145

    def verify_decomp(U, V, W, n, m, p):
        """Verify tensor decomposition. U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2_kernels():
        """Return the 7 Strassen terms for 2x2*2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: P1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # W[0,0] for C00, W[1,1] for C11. Index 0 is k=0,i=0. Index 3 is k=1,i=1.
        
        # Term 1: P2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # W[1,0] for C10 (idx 2), W[1,1] for C11 (idx 3)
        
        # Term 2: P3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # W[0,1] for C01 (idx 1), W[1,1] for C11 (idx 3)
        
        # Term 3: P4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1 # W[0,0] for C00 (idx 0), W[1,1] for C11 (idx 3)
        
        # Term 4: P5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # W[0,0] for C00 (idx 0), W[0,1] for C01 (idx 1)
        
        # Term 5: P6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1 # W[1,1] for C11 (idx 3)
        
        # Term 6: P7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # W[0,0] for C00 (idx 0)
        
        return U, V, W

    U2, V2, W2 = strassen_2x2_kernels()
    
    # Build 4x4x4 part (rank 49)
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U2[i1*2+j1, t1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val2 = U2[i2*2+j2, t2]
                            if val2 != 0:
                                row = (2*i1+i2)*4 + (2*j1+j2)
                                U44[row, col] = val1 * val2
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V2[j1*2+k1, t1]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            val2 = V2[j2*2+k2, t2]
                            if val2 != 0:
                                row = (2*j1+j2)*4 + (2*k1+k2)
                                V44[row, col] = val1 * val2
            for k1 in range(2):
                for i1 in range(2):
                    val1 = W2[k1*2+i1, t1]
                    if val1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            val2 = W2[k2*2+i2, t2]
                            if val2 != 0:
                                row = (2*k1+k2)*4 + (2*i1+i2)
                                W44[row, col] = val1 * val2

    # Build full 4x4x5 (rank 50)
    rank_total = 50
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, :49] = U44
    V[:16, :49] = V44
    W[:16, :49] = W44
    
    # Term 49: 5th column
    U[:, 49] = 1.0
    # V[j*5+4, 49] = 1 for j=0,1,2,3 -> indices 4, 9, 14, 19
    V[[4, 9, 14, 19], 49] = 1.0
    # W[4*4+i, 49] = 1 for i=0,1,2,3 -> indices 16, 17, 18, 19
    W[[16, 17, 18, 19], 49] = 1.0
    
    if verify_decomp(U, V, W, 4, 4, 5):
        if time.time() < deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook rank 80
    rank = 80
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i*4+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*4+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
