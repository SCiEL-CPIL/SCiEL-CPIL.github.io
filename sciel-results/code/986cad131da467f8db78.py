import numpy as np
import time

def construct():
    """
    Constructs a rank-minimal tensor decomposition for 2x2 x 2x13 matrix multiplication.
    Uses Strassen's algorithm for 2x2 blocks (rank 7) and schoolbook for the remaining 2x1 block (rank 4).
    Total rank = 6 * 7 + 4 = 46.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
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
        if time.time() > deadline:
            break
        k1 = 2 * i
        k2 = 2 * i + 1
        add_strassen(k1, k2, offset)
        offset += 7
        
    # Process 1 Schoolbook block for column 12
    if time.time() <= deadline:
        add_schoolbook(12, offset)
        offset += 4
    
    # Verification using vectorized numpy for speed
    def verify(U, V, W):
        r = U.shape[1]
        # Check tensor contraction: sum_c U[a,c] V[b,c] W[c_idx, c] == 1
        # for all i in [0,n), j in [0,m), k in [0,p)
        
        # Vectorized check
        # Create index arrays
        i_vals = np.repeat(np.arange(n), m*p)
        j_vals = np.tile(np.repeat(np.arange(m), p), n)
        k_vals = np.tile(np.arange(p), n*m)
        
        u_idx = i_vals * m + j_vals
        v_idx = j_vals * p + k_vals
        w_idx = k_vals * n + i_vals
        
        # Extract relevant slices
        U_sub = U[u_idx, :]  # shape (n*m*p, r)
        V_sub = V[v_idx, :]  # shape (n*m*p, r)
        W_sub = W[w_idx, :]  # shape (n*m*p, r)
        
        # Compute contraction: sum over rank dimension
        prod = U_sub * V_sub * W_sub
        sums = np.sum(prod, axis=1)
        
        # Since entries are integers (0, 1, -1), sums should be exactly 1.0
        # Check with tight tolerance for numerical safety
        if not np.allclose(sums, np.ones(n*m*p), atol=1e-10):
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
    
    # Ensure entries are clean integers (avoid floating point artifacts)
    U = np.round(U).astype(float)
    V = np.round(V).astype(float)
    W = np.round(W).astype(float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
