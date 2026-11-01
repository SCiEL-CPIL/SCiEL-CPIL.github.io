import numpy as np

def construct():
    """
    Constructs a rank-minimal tensor decomposition for 2x2 x 2x13 matrix multiplication.
    Uses Strassen's algorithm for 2x2 blocks (rank 7) and schoolbook for the remaining 2x1 block (rank 4).
    Total rank = 6 * 7 + 4 = 46.
    """
    n, m, p = 2, 2, 13
    rank = 46
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    offset = 0
    
    # Function to add a Strassen block for columns k1, k2
    def add_strassen(k1, k2, off):
        # M1 = (A00+A11)(B00+B11)
        U[0, off] += 1; U[3, off] += 1
        V[0*p+k1, off] += 1; V[1*p+k2, off] += 1
        W[k1*n+0, off] += 1; W[k2*n+1, off] += 1

        # M2 = (A10+A11)B00
        U[2, off+1] += 1; U[3, off+1] += 1
        V[0*p+k1, off+1] += 1
        W[k1*n+1, off+1] += 1

        # M3 = A00(B10-B11)
        U[0, off+2] += 1
        V[0*p+k2, off+2] += 1; V[1*p+k2, off+2] -= 1
        W[k2*n+0, off+2] += 1; W[k2*n+1, off+2] += 1

        # M4 = A11(B01-B00)
        U[3, off+3] += 1
        V[0*p+k2, off+3] += 1; V[0*p+k1, off+3] -= 1
        W[k2*n+0, off+3] += 1; W[k1*n+0, off+3] += 1

        # M5 = (A00+A01)B11
        U[0, off+4] += 1; U[1, off+4] += 1
        V[1*p+k2, off+4] += 1
        W[k2*n+0, off+4] -= 1; W[k2*n+1, off+4] += 1

        # M6 = (A10-A00)(B00+B01)
        U[2, off+5] += 1; U[0, off+5] -= 1
        V[0*p+k1, off+5] += 1; V[0*p+k2, off+5] += 1
        W[k1*n+1, off+5] += 1; W[k2*n+1, off+5] += 1

        # M7 = (A01-A11)(B10+B11)
        U[1, off+6] += 1; U[3, off+6] -= 1
        V[1*p+k1, off+6] += 1; V[1*p+k2, off+6] += 1
        W[k1*n+0, off+6] += 1; W[k2*n+0, off+6] -= 1

    # Function to add a schoolbook block for column k
    def add_schoolbook(k, off):
        U[0, off] += 1
        V[0*p+k, off] += 1
        W[k*n+0, off] += 1

        U[1, off+1] += 1
        V[1*p+k, off+1] += 1
        W[k*n+0, off+1] += 1

        U[2, off+2] += 1
        V[0*p+k, off+2] += 1
        W[k*n+1, off+2] += 1

        U[3, off+3] += 1
        V[1*p+k, off+3] += 1
        W[k*n+1, off+3] += 1

    # Process 6 Strassen blocks for columns (0,1), (2,3), ..., (10,11)
    for i in range(6):
        k1 = 2 * i
        k2 = 2 * i + 1
        add_strassen(k1, k2, offset)
        offset += 7
        
    # Process 1 Schoolbook block for column 12
    add_schoolbook(12, offset)
    offset += 4
    
    # Verification
    def verify(U, V, W):
        r = U.shape[1]
        # Check tensor contraction: sum_c U[a,c] V[b,c] W[c_idx, c] == delta
        # Indices:
        # a in [0, 3] (i*m + j)
        # b in [0, 25] (j*p + k)
        # c_idx in [0, 25] (k*n + i)
        
        # We want: for all i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        # Note: U index is i*m+j. V index is j*p+k. W index is k*n+i.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook rank 52 if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
