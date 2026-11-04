"""
Rank-49 tensor decomposition for 4x4x4 matrix multiplication.
Uses the Kronecker product of the Strassen rank-7 decomposition for 2x2x2.

Tensor convention:
  U: shape (n*m, rank), U[i*m + j, r] = coefficient of A[i,j] in rank-r term
  V: shape (m*p, rank), V[j*p + k, r] = coefficient of B[j,k] in rank-r term
  W: shape (p*n, rank), W[k*n + i, r] = coefficient of C[i,k] in rank-r term

The decomposition satisfies:
  T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
where T is the matrix multiplication tensor (T[i,j,k] = 1 iff the
(i,j) entry of A multiplies the (j,k) entry of B to contribute to
the (i,k) entry of C, i.e., the index j is shared).
"""

import time
import numpy as np


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    U: shape (4, 7), indexed by (i*2 + j), i,j in {0,1}
    V: shape (4, 7), indexed by (j*2 + k), j,k in {0,1}
    W: shape (4, 7), indexed by (k*2 + i), k,i in {0,1}
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Strassen 7 intermediate products for C = A * B:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    U[0][0] = 1.0; U[3][0] = 1.0    # A00, A11
    V[0][0] = 1.0; V[3][0] = 1.0    # B00, B11
    W[0][0] = 1.0; W[3][0] = 1.0    # C00, C11
    
    # Term 2: M2
    U[2][1] = 1.0; U[3][1] = 1.0    # A10, A11
    V[0][1] = 1.0                    # B00
    W[2][1] = 1.0; W[3][1] = -1.0   # C10, C11
    
    # Term 3: M3
    U[0][2] = 1.0                    # A00
    V[1][2] = 1.0; V[3][2] = -1.0   # B01, B11
    W[1][2] = 1.0; W[3][2] = 1.0    # C01, C11
    
    # Term 4: M4
    U[3][3] = 1.0                    # A11
    V[2][3] = 1.0; V[0][3] = -1.0   # B10, B00
    W[0][3] = 1.0; W[2][3] = 1.0    # C00, C10
    
    # Term 5: M5
    U[0][4] = 1.0; U[1][4] = 1.0    # A00, A01
    V[3][4] = 1.0                    # B11
    W[0][4] = -1.0; W[1][4] = 1.0   # C00, C01
    
    # Term 6: M6
    U[2][5] = 1.0; U[0][5] = -1.0   # A10, A00
    V[0][5] = 1.0; V[1][5] = 1.0    # B00, B01
    W[3][5] = 1.0                    # C11
    
    # Term 7: M7
    U[1][6] = 1.0; U[3][6] = -1.0   # A01, A11
    V[2][6] = 1.0; V[3][6] = 1.0    # B10, B11
    W[0][6] = 1.0                    # C00
    
    return U, V, W


def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    
    For 4x4x4 with n=m=p=4:
    i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    
    Kronecker structure: r = r1 * 7 + r2
    U4[i*m+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    V4[j*p+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    W4[k*n+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
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
                                    
                                    u_val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                    v_val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                    w_val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                    
                                    U[i * m + j][r] = u_val
                                    V[j * p + k][r] = v_val
                                    W[k * n + i][r] = w_val
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] = 1 if and only if the (i,j) entry of A times (j,k) entry of B
    contributes to (i,k) entry of C. This is the standard matmul tensor.
    
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i,j,k]
    """
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
                if err > 1e-10:
                    return False, max_err, (i, j, k)
    return True, max_err, None


def construct():
    n, m, p = 4, 4, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    # Build the rank-49 Kronecker decomposition
    result = _kronecker_decomposition(n, m, p)
    
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    # Verify correctness
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        valid, max_err, coord = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # If verification fails, fall back to schoolbook rank-64
            rank = n * m * p
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1.0
                        V[j * p + k][col] = 1.0
                        W[k * n + i][col] = 1.0
                        col += 1
    
    return {"U": U, "V": V, "W": W}
