"""
Program for 4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of two Strassen rank-7 decompositions to achieve rank 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    Indexing: A[i,j] -> index i*2+j, B[j,k] -> index j*2+k, C[i,k] -> index i*2+k
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A11(B10-B00) -> C10 += M4, C01 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11 -> C00 -= M5, C10 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
    equals 1 if (i,j,k) is a valid triple, else 0.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_diff = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c]))
                diff = abs(val - 1.0)
                if diff > max_diff:
                    max_diff = diff
    # Also check that off-diagonal entries are zero
    # Sample a few off-diagonal entries to keep it fast
    for _ in range(50):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        # Check if this is a valid (i,j,k) triple
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (i == i2 and j == j2 and k == k2):
            val = float(np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c]))
            diff = abs(val)
            if diff > max_diff:
                max_diff = diff
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            for k1 in range(2):
                                for k2 in range(2):
                                    i = 2 * i1 + i2
                                    j = 2 * j1 + j2
                                    k = 2 * k1 + k2
                                    a = i * m + j
                                    b = j * p + k
                                    c = k * n + i
                                    U[a][r] += U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                    V[b][r] += V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                    W[c][r] += W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

# (Removed unused Laderman and embedding functions to reduce code size)

def _schoolbook_decomposition(n, m, p):
    """Construct the rank-64 schoolbook decomposition for n x m x p."""
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
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline with 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use Kronecker-Strassen rank-49 decomposition
    try:
        result49 = _kronecker_decomposition(n, m, p)
        
        # Verify the decomposition
        U, V, W = result49["U"], result49["V"], result49["W"]
        valid, max_diff = _verify_decomp(n, m, p, U, V, W, tol=1e-9)
        
        if valid:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Check deadline before expensive fallback
    if time.time() > deadline:
        pass
    
    # Final fallback: schoolbook
    result_schoolbook = _schoolbook_decomposition(n, m, p)
    return result_schoolbook

# EVOLVE-BLOCK-END
