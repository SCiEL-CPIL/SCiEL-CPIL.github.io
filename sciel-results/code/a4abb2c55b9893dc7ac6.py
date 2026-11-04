"""
Matrix multiplication 4,4,4 tensor decomposition.
Uses the Kronecker product of Strassen's rank-7 algorithm for 2x2 matrices,
yielding a rank-49 decomposition for 4x4 matrices.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
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
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2 * j1 + j2
                            
                            idx_u = i * m + k
                            idx_v = k * p + j
                            idx_w = j * n + i
                            
                            idx_u2_1 = i1 * 2 + k1
                            idx_u2_2 = i2 * 2 + k2
                            idx_v2_1 = k1 * 2 + j1
                            idx_v2_2 = k2 * 2 + j2
                            idx_w2_1 = j1 * 2 + i1
                            idx_w2_2 = j2 * 2 + i2
                            
                            for r1 in range(7):
                                u1_val = U2[idx_u2_1][r1]
                                v1_val = V2[idx_v2_1][r1]
                                w1_val = W2[idx_w2_1][r1]
                                
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    u2_val = U2[idx_u2_2][r2]
                                    v2_val = V2[idx_v2_2][r2]
                                    w2_val = W2[idx_w2_2][r2]
                                    
                                    U[idx_u][r] = u1_val * u2_val
                                    V[idx_v][r] = v1_val * v2_val
                                    W[idx_w][r] = w1_val * w2_val
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _schoolbook_decomposition(n, m, p):
    """Fallback rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We use the Rank 49 Kronecker-Strassen decomposition.
    try:
        result = _kronecker_decomposition(n, m, p)
        
        # Verify correctness
        valid, max_diff = _verify_decomp(n, m, p, result["U"], result["V"], result["W"])
        
        if valid:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
        else:
            fallback = _schoolbook_decomposition(n, m, p)
            return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}
            
    except Exception:
        fallback = _schoolbook_decomposition(n, m, p)
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}
# EVOLVE-BLOCK-END
