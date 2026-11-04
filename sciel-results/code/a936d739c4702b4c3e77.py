"""
Program for 4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen rank-7 with runtime verification
to guarantee correct index mapping against the evaluator's trace convention.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, r] = coefficient of A[i,j] in term r
    V[j*2+k, r] = coefficient of B[j,k] in term r
    W[k*2+i, r] = coefficient of C[i,k] in term r (trace convention)
    """
    rank = 7
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((4, rank), dtype=int)
    W = np.zeros((4, rank), dtype=int)
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
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


def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    
    Index convention:
    - A is n x m: A[i,j], i in [0,n), j in [0,m)
    - B is m x p: B[j,k], j in [0,m), k in [0,p)
    - C is n x p: C[i,k], i in [0,n), k in [0,p)
    - U flat index: i*m + j
    - V flat index: j*p + k
    - W flat index: k*n + i (trace convention: W[k*n+i, r])
    
    Tensor: T[i*m+j, j*p+k, k*n+i] = 1
    
    Kronecker: i=2*i1+i2, j=2*j1+j2, k=2*k1+k2
    U[i*m+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    V[j*p+k, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    W[k*n+i, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                                    V[j * p + k, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    W[k * n + i, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill
    
    # Primary: Kronecker product of Strassen (rank 49)
    try:
        result49 = _kronecker_decomposition(n, m, p)
        valid49, diff49 = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"])
        
        if valid49:
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
        else:
            # If verification fails, check deadline and fall back
            if time.time() > deadline:
                return _schoolbook(n, m, p)
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
