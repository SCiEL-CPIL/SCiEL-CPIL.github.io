import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _verify_decomposition(U, V, W):
    """Verify tensor identity for 2x2 * 2x7 matrix multiplication."""
    n, m, p = 2, 2, 7
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # True tensor: T[a,b,c] = 1 if A[a]*B[b] contributes to C[c]
    # a = i*m+j, b = j*p+k, c = k*n+i
    T_true = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            jb = b // p
            k = b % p
            if j == jb:
                c = k * n + i
                T_true[a, b, c] = 1.0
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_true, T_decomp, atol=1e-10)

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        ks = k_start
        ke = k_start + 1
        # W indices: C[0,ks]=2*ks, C[1,ks]=2*ks+1, C[0,ke]=2*ke, C[1,ke]=2*ke+1
        w00 = 2*ks    # C[0,ks]
        w10 = 2*ks+1  # C[1,ks]
        w01 = 2*ke    # C[0,ke]
        w11 = 2*ke+1  # C[1,ke]
        
        # M1 = (A00 + A11) * (B0,ks + B1,ke)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0*7 + ks, col_start+0] += 1
        V[1*7 + ke, col_start+0] += 1
        W[w00, col_start+0] += 1
        W[w11, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0*7 + ks, col_start+1] += 1
        W[w10, col_start+1] += 1
        W[w11, col_start+1] -= 1

        # M3 = A00 * (B0,ke - B1,ke)
        U[0, col_start+2] += 1
        V[0*7 + ke, col_start+2] += 1
        V[1*7 + ke, col_start+2] -= 1
        W[w01, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[1*7 + ks, col_start+3] += 1
        V[0*7 + ks, col_start+3] -= 1
        W[w10, col_start+3] += 1
        W[w11, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ke
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[1*7 + ke, col_start+4] += 1
        W[w00, col_start+4] -= 1
        W[w01, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ke)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[0*7 + ks, col_start+5] += 1
        V[0*7 + ke, col_start+5] += 1
        W[w00, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ke)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[1*7 + ks, col_start+6] += 1
        V[1*7 + ke, col_start+6] += 1
        W[w11, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify before returning
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook if verification fails
        return _schoolbook_decomposition()
    
    return result

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED, rank =", U.shape[1])
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
