"""
Rank-77 exact tensor decomposition for 4x4 x 4x6 matrix multiplication.
Uses Kronecker product of Strassen 2x2x2 (rank 7) and 2x2x3 (rank 11).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition (rank 7).
    Indices: row = i*2 + j for U/V (where j is col index of B in 2x2), 
    W index row = k*2 + i.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_2x2x3_hybrid():
    """
    Returns U, V, W for 2x2x3 multiplication.
    U: 4x11, V: 6x11, W: 6x11.
    First 7 columns are Strassen for cols 0,1.
    Last 4 columns are schoolbook for col 2.
    """
    U1, V1, W1 = _get_strassen_2x2()
    
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float) # 2 rows, 3 cols -> 6 entries
    W2 = np.zeros((6, 11), dtype=float) # 3 rows, 2 cols -> 6 entries (k*2+i)
    
    # Copy Strassen part (cols 0,1 of B map to 0,1 and 3,4 in V2; 0,1,2,3 in W2)
    # B indices in V2: row = j*3 + k.
    # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 3; j=1, k=1 -> 4.
    # W indices in W2: row = k*2 + i.
    # k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3.
    
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        
    # Schoolbook for k=2 (col 2 of B)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0]B[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1 # j=0, k=2 -> 0*3+2=2
    W2[4, 7] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 8: A[0,1]B[1,2]
    U2[1, 8] = 1
    V2[5, 8] = 1 # j=1, k=2 -> 1*3+2=5
    W2[4, 8] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 9: A[1,0]B[0,2]
    U2[2, 9] = 1
    V2[2, 9] = 1 # j=0, k=2 -> 0*3+2=2
    W2[5, 9] = 1 # k=2, i=1 -> 2*2+1=5
    
    # Term 10: A[1,1]B[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1 # j=1, k=2 -> 1*3+2=5
    W2[5, 10] = 1 # k=2, i=1 -> 2*2+1=5
    
    return U2, V2, W2

def _kron_decomposition(n, m, p):
    """
    Constructs the full 4x4x6 decomposition using Kronecker products of 2x2x2 and 2x2x3.
    """
    U1, V1, W1 = _get_strassen_2x2()
    U2, V2, W2 = _get_2x2x3_hybrid()
    
    rank1 = U1.shape[1] # 7
    rank2 = U2.shape[1] # 11
    total_rank = rank1 * rank2 # 77
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    # Precompute nonzero indices for speed
    U1_nz = {c: np.nonzero(U1[:, c])[0] for c in range(rank1)}
    U2_nz = {c: np.nonzero(U2[:, c])[0] for c in range(rank2)}
    V1_nz = {c: np.nonzero(V1[:, c])[0] for c in range(rank1)}
    V2_nz = {c: np.nonzero(V2[:, c])[0] for c in range(rank2)}
    W1_nz = {c: np.nonzero(W1[:, c])[0] for c in range(rank1)}
    W2_nz = {c: np.nonzero(W2[:, c])[0] for c in range(rank2)}
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            
            # U: (i1, j1) from U1, (i2, j2) from U2
            u1_vals = U1_nz[c1]
            u2_vals = U2_nz[c2]
            for idx1 in u1_vals:
                val1 = U1[idx1, c1]
                i1 = idx1 // 2
                j1 = idx1 % 2
                for idx2 in u2_vals:
                    val2 = U2[idx2, c2]
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    row = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    U[row, c] = val1 * val2
            
            # V: (j1, k1) from V1, (j2, k2) from V2
            v1_vals = V1_nz[c1]
            v2_vals = V2_nz[c2]
            for idx1 in v1_vals:
                val1 = V1[idx1, c1]
                j1 = idx1 // 2
                k1 = idx1 % 2
                for idx2 in v2_vals:
                    val2 = V2[idx2, c2]
                    j2 = idx2 // 3
                    k2 = idx2 % 3
                    row = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                    V[row, c] = val1 * val2
            
            # W: (k1, i1) from W1, (k2, i2) from W2
            w1_vals = W1_nz[c1]
            w2_vals = W2_nz[c2]
            for idx1 in w1_vals:
                val1 = W1[idx1, c1]
                k1 = idx1 // 2
                i1 = idx1 % 2
                for idx2 in w2_vals:
                    val2 = W2[idx2, c2]
                    k2 = idx2 // 2
                    i2 = idx2 % 2
                    row = (k1 * 3 + k2) * 4 + (i1 * 2 + i2)
                    W[row, c] = val1 * val2
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies the decomposition using random test cases.
    """
    np.random.seed(12345)
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            alpha = np.sum(A * U[:, r].reshape(n, m))
            beta = np.sum(B * V[:, r].reshape(m, p))
            if alpha != 0 and beta != 0:
                C_computed += alpha * beta * W[:, r].reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook_fallback(n, m, p):
    """Returns schoolbook rank n*m*p decomposition as fallback."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    try:
        U, V, W = _kron_decomposition(n, m, p)
        
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded during construction")
            
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            return _schoolbook_fallback(n, m, p)
    except Exception:
        return _schoolbook_fallback(n, m, p)
# EVOLVE-BLOCK-END
