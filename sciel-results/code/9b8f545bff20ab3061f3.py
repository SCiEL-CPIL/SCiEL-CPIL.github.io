import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _time_remaining():
    return _DEADLINE - time.time()

def _verify_decomposition(U, V, W, n=2, m=2, p=7, trials=5, seed=12345):
    """Quick verification with random integer matrices."""
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, (n, m)).astype(float)
        B = rng.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_W = np.zeros(p * n)
        for r in range(U.shape[1]):
            SA = np.dot(U[:, r], A.flatten())
            SB = np.dot(V[:, r], B.flatten())
            C_W += W[:, r] * SA * SB
        C_mat = C_W.reshape(p, n).T
        if not np.allclose(C_mat, C_expected, atol=1e-9):
            return False
    return True

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses 3 Strassen blocks (rank 7 each) for columns 0-5, and schoolbook (rank 4) for column 6.
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(col_start, k_start):
        """
        Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1.
        
        A indexing: A[0]=A[0,0], A[1]=A[0,1], A[2]=A[1,0], A[3]=A[1,1]
        B indexing: B[j*p+k] for row j, column k
        W indexing: W[k*n+i] for output C[i,k]
        """
        # M1 = (A00 + A11) * (B[0,k0] + B[1,k1])
        # C[0,k0] += M1, C[1,k1] += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0*p + k_start, col_start+0] += 1
        V[1*p + (k_start+1), col_start+0] += 1
        W[k_start*n + 0, col_start+0] += 1
        W[(k_start+1)*n + 1, col_start+0] += 1
        
        # M2 = (A10 + A11) * B[0,k0]
        # C[1,k0] += M2, C[1,k1] -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0*p + k_start, col_start+1] += 1
        W[k_start*n + 1, col_start+1] += 1
        W[(k_start+1)*n + 1, col_start+1] -= 1
        
        # M3 = A00 * (B[0,k1] - B[1,k1])
        # C[0,k1] += M3, C[1,k1] += M3
        U[0, col_start+2] += 1
        V[0*p + (k_start+1), col_start+2] += 1
        V[1*p + (k_start+1), col_start+2] -= 1
        W[(k_start+1)*n + 0, col_start+2] += 1
        W[(k_start+1)*n + 1, col_start+2] += 1
        
        # M4 = A11 * (B[1,k0] - B[0,k0])
        # C[0,k0] += M4, C[1,k0] += M4
        U[3, col_start+3] += 1
        V[1*p + k_start, col_start+3] += 1
        V[0*p + k_start, col_start+3] -= 1
        W[k_start*n + 0, col_start+3] += 1
        W[k_start*n + 1, col_start+3] += 1
        
        # M5 = (A00 + A01) * B[1,k1]
        # C[0,k0] -= M5, C[0,k1] += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[1*p + (k_start+1), col_start+4] += 1
        W[k_start*n + 0, col_start+4] -= 1
        W[(k_start+1)*n + 0, col_start+4] += 1
        
        # M6 = (A10 - A00) * (B[0,k0] + B[0,k1])
        # C[1,k1] += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[0*p + k_start, col_start+5] += 1
        V[0*p + (k_start+1), col_start+5] += 1
        W[(k_start+1)*n + 1, col_start+5] += 1
        
        # M7 = (A01 - A11) * (B[1,k0] + B[1,k1])
        # C[0,k0] += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[1*p + k_start, col_start+6] += 1
        V[1*p + (k_start+1), col_start+6] += 1
        W[k_start*n + 0, col_start+6] += 1
    
    # Add three Strassen blocks for columns (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Schoolbook for column 6 (rank 4)
    col_start = 21
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    U[0, col_start+0] += 1
    V[0*p + 6, col_start+0] += 1
    W[6*n + 0, col_start+0] += 1

    U[1, col_start+1] += 1
    V[1*p + 6, col_start+1] += 1
    W[6*n + 0, col_start+1] += 1

    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    U[2, col_start+2] += 1
    V[0*p + 6, col_start+2] += 1
    W[6*n + 1, col_start+2] += 1

    U[3, col_start+3] += 1
    V[1*p + 6, col_start+3] += 1
    W[6*n + 1, col_start+3] += 1

    # Self-verify before returning
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: pure schoolbook rank 28
        U2 = np.zeros((n*m, 28))
        V2 = np.zeros((m*p, 28))
        W2 = np.zeros((p*n, 28))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m + j + (j*p + k)*0  # placeholder
                    # Use flat indexing: term index = (i*n + k)*m + j
                    t = (i*n + k)*m + j
                    U2[i*m + j, t] = 1
                    V2[j*p + k, t] = 1
                    W2[k*n + i, t] = 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
    print(f"Verification: {'PASSED' if _verify_decomposition(U, V, W) else 'FAILED'}")
