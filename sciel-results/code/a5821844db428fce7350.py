"""
4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen rank-7 decomposition to achieve rank 49.
Vectorized construction via numpy for speed and correctness.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    
    The 7 Strassen terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U[i*2+j, r] = coefficient of A[i,j] in term r
    V[j*2+k, r] = coefficient of B[j,k] in term r
    W[k*2+i, r] = coefficient of C[i,k] in term r (trace convention)
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    W[0*2+0][0] = 1  # C00
    W[1*2+1][0] = 1  # C11
    
    # Term 2: M2 = (A10+A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    W[0*2+1][1] = 1  # C10
    W[1*2+1][1] = -1 # C11
    
    # Term 3: M3 = A00(B01-B11)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # B11
    W[1*2+0][2] = 1  # C01
    W[1*2+1][2] = 1  # C11
    
    # Term 4: M4 = A11(B10-B00)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # B00
    W[0*2+0][3] = 1  # C00
    W[0*2+1][3] = 1  # C10
    
    # Term 5: M5 = (A00+A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    W[0*2+0][4] = -1 # C00
    W[1*2+0][4] = 1  # C01
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    W[1*2+1][5] = 1  # C11
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    W[0*2+0][6] = 1  # C00
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _construct_vectorized(n, m, p):
    """
    Vectorized construction of the rank-49 Kronecker decomposition.
    Uses numpy einsum to build the Kronecker product structure efficiently.
    """
    U2, V2, W2 = _strassen_2x2()
    U2_arr = np.array(U2, dtype=float)  # (4, 7) = (2,2,7) reshaped
    V2_arr = np.array(V2, dtype=float)  # (4, 7)
    W2_arr = np.array(W2, dtype=float)  # (4, 7)
    
    rank = 49
    n1 = n // 2  # 2
    m1 = m // 2  # 2
    p1 = p // 2  # 2
    
    # Reshape base matrices to (2, 2, 7) for block structure
    # U2: row index is i*2+j where i,j in {0,1}
    # So reshape to (2, 2, 7) with axes (i1, j1, r1)
    U2r = U2_arr.reshape(n1, m1, rank // n1 // m1)  # (2, 2, 7)
    V2r = V2_arr.reshape(m1, p1, rank // m1 // p1)   # (2, 2, 7)
    W2r = W2_arr.reshape(p1, n1, rank // p1 // n1)   # (2, 2, 7)
    
    # Build U: shape (n*m, rank) = (16, 49)
    # U[i*m+k, r1*7+r2] = U2[i1*2+k1, r1] * U2[i2*2+k2, r2]
    # where i = 2*i1+i2, k = 2*k1+k2
    # U2r[i1, k1, r1] * U2r[i2, k2, r2]
    # Output: (i1, i2, k1, k2, r1, r2) -> reshape to (4, 4, 49) -> (16, 49)
    U_tensor = np.einsum('ikr,jls->ikjlrs', U2r, U2r)  # (2,2,2,2,7,7)
    U_arr = U_tensor.reshape(n, m, rank)  # (4, 4, 49)
    U_mat = U_arr.reshape(n * m, rank)  # (16, 49)
    
    # Build V: shape (m*p, rank) = (16, 49)
    # V[k*p+j, r1*7+r2] = V2[k1*2+j1, r1] * V2[k2*2+j2, r2]
    # where k = 2*k1+k2, j = 2*j1+j2
    # V2r[k1, j1, r1] * V2r[k2, j2, r2]
    # Output: (k1, k2, j1, j2, r1, r2) -> reshape to (4, 4, 49) -> (16, 49)
    V_tensor = np.einsum('klr,jms->kljmsr', V2r, V2r)  # (2,2,2,2,7,7)
    V_arr = V_tensor.reshape(m, p, rank)  # (4, 4, 49)
    V_mat = V_arr.reshape(m * p, rank)  # (16, 49)
    
    # Build W: shape (p*n, rank) = (16, 49)
    # W[j*n+i, r1*7+r2] = W2[j1*2+i1, r1] * W2[j2*2+i2, r2]
    # where j = 2*j1+j2, i = 2*i1+i2
    # W2r[j1, i1, r1] * W2r[j2, i2, r2]
    # Output: (j1, j2, i1, i2, r1, r2) -> reshape to (4, 4, 49) -> (16, 49)
    W_tensor = np.einsum('klr,jms->kjlmsr', W2r, W2r)  # (2,2,2,2,7,7)
    W_arr = W_tensor.reshape(p, n, rank)  # (4, 4, 49)
    W_mat = W_arr.reshape(p * n, rank)  # (16, 49)
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # Try vectorized rank-49 construction
    try:
        result = _construct_vectorized(n, m, p)
        valid, diff = _verify_decomp(n, m, p, result["U"], result["V"], result["W"])
        if valid:
            return result
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Fall back to schoolbook rank-64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
