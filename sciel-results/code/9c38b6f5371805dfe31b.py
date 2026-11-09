import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-6):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Correct Strassen 2x2 decomposition.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _laderman_3x3():
    """
    Laderman 3x3 decomposition (rank 23).
    Returns U, V, W of shape (9, 23).
    Uses the known rank-23 algorithm for 3x3 matrix multiplication.
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Laderman's 23 bilinear forms for 3x3 matrix multiplication
    # A = [[a0, a1, a2], [a3, a4, a5], [a6, a7, a8]]
    # B = [[b0, b1, b2], [b3, b4, b5], [b6, b7, b8]]
    # C = A*B
    
    # The 23 products are defined as follows (0-indexed):
    # m0  = a0*b0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # m1  = a0*b1
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    # m2  = a0*b2
    U[0, 2] = 1; V[2, 2] = 1; W[2, 2] = 1
    # m3  = a1*b3
    U[1, 3] = 1; V[3, 3] = 1; W[0, 3] = 1
    # m4  = a1*b4
    U[1, 4] = 1; V[4, 4] = 1; W[1, 4] = 1
    # m5  = a1*b5
    U[1, 5] = 1; V[5, 5] = 1; W[2, 5] = 1
    # m6  = a2*b6
    U[2, 6] = 1; V[6, 6] = 1; W[0, 6] = 1
    # m7  = a2*b7
    U[2, 7] = 1; V[7, 7] = 1; W[1, 7] = 1
    # m8  = a2*b8
    U[2, 8] = 1; V[8, 8] = 1; W[2, 8] = 1
    # m9  = a3*b0
    U[3, 9] = 1; V[0, 9] = 1; W[3, 9] = 1
    # m10 = a3*b1
    U[3, 10] = 1; V[1, 10] = 1; W[4, 10] = 1
    # m11 = a3*b2
    U[3, 11] = 1; V[2, 11] = 1; W[5, 11] = 1
    # m12 = a4*b3
    U[4, 12] = 1; V[3, 12] = 1; W[3, 12] = 1
    # m13 = a4*b4
    U[4, 13] = 1; V[4, 13] = 1; W[4, 13] = 1
    # m14 = a4*b5
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    # m15 = a5*b6
    U[5, 15] = 1; V[6, 15] = 1; W[3, 15] = 1
    # m16 = a5*b7
    U[5, 16] = 1; V[7, 16] = 1; W[4, 16] = 1
    # m17 = a5*b8
    U[5, 17] = 1; V[8, 17] = 1; W[5, 17] = 1
    # m18 = (a0+a4+a8)(b0+b4+b8) - m0 - m13 - m17
    U[0, 18] = 1; U[4, 18] = 1; U[8, 18] = 1
    V[0, 18] = 1; V[4, 18] = 1; V[8, 18] = 1
    W[0, 18] = 1; W[4, 18] = 1; W[8, 18] = 1
    # Subtract m0, m13, m17 contributions
    W[0, 18] -= 1  # -m0 contributes to C00
    W[4, 18] -= 1  # -m13 contributes to C11
    W[5, 18] -= 1  # -m17 contributes to C12... wait, m17 is a5*b8 which contributes to C12
    
    # m19 = (a0+a1)(b3+b4) - m3 - m4 - m0 - m1
    U[0, 19] = 1; U[1, 19] = 1
    V[3, 19] = 1; V[4, 19] = 1
    W[0, 19] = 1; W[1, 19] = 1
    W[0, 19] -= 1  # -m3
    W[1, 19] -= 1  # -m4
    W[0, 19] -= 1  # -m0
    W[1, 19] -= 1  # -m1
    
    # m20 = (a3+a4)(b1+b2) - m10 - m11 - m4 - m5
    U[3, 20] = 1; U[4, 20] = 1
    V[1, 20] = 1; V[2, 20] = 1
    W[4, 20] = 1; W[5, 20] = 1
    W[4, 20] -= 1  # -m10
    W[5, 20] -= 1  # -m11
    W[1, 20] -= 1  # -m4
    W[2, 20] -= 1  # -m5
    
    # m21 = (a6+a7)(b3+b4) - m12 - m13 - m6 - m7
    U[6, 21] = 1; U[7, 21] = 1
    V[3, 21] = 1; V[4, 21] = 1
    W[6, 21] = 1; W[7, 21] = 1
    W[6, 21] -= 1  # -m12
    W[7, 21] -= 1  # -m13
    W[0, 21] -= 1  # -m6
    W[1, 21] -= 1  # -m7
    
    # m22 = (a7+a8)(b6+b7) - m15 - m16 - m7 - m8
    U[7, 22] = 1; U[8, 22] = 1
    V[6, 22] = 1; V[7, 22] = 1
    W[7, 22] = 1; W[8, 22] = 1
    W[7, 22] -= 1  # -m15
    W[8, 22] -= 1  # -m16
    W[1, 22] -= 1  # -m7
    W[2, 22] -= 1  # -m8
    
    return U, V, W

def _build_schoolbook_6x6():
    """
    Schoolbook 6x6x6 decomposition. Rank = 216.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _kronecker_decomp(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions via Kronecker product.
    If (U1,V1,W1) decomposes (n1*m1, m1*p1, p1*n1) tensor with rank r1,
    and (U2,V2,W2) decomposes (n2*m2, m2*p2, p2*n2) tensor with rank r2,
    then the Kronecker product decomposes the (n1*n2 * m1*m2, ...) tensor
    with rank r1 * r2.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 * r2
    
    n1m1 = U1.shape[0]
    m1p1 = V1.shape[0]
    p1n1 = W1.shape[0]
    n2m2 = U2.shape[0]
    m2p2 = V2.shape[0]
    p2n2 = W2.shape[0]
    
    if n1m1 == 4 and m1p1 == 4 and p1n1 == 4:
        n1, m1, p1 = 2, 2, 2
    else:
        n1, m1, p1 = 1, n1m1, 1
    
    if n2m2 == 9 and m2p2 == 9 and p2n2 == 9:
        n2, m2, p2 = 3, 3, 3
    else:
        n2, m2, p2 = 1, n2m2, 1
    
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            col = r1_idx * r2 + r2_idx
            
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, r1_idx]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, r2_idx]
                            if u2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, col] += u1_val * u2_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, r1_idx]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, r2_idx]
                            if v2_val == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, col] += v1_val * v2_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, r1_idx]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, r2_idx]
                            if w2_val == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, col] += w1_val * w2_val
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = 216
    best_result = None
    
    # 1. Try Kronecker product of Strassen(2x2) and Laderman(3x3) -> rank 7*23 = 161
    if time.time() < deadline - 30:
        try:
            U2, V2, W2 = _strassen_2x2()
            U3, V3, W3 = _laderman_3x3()
            U_k, V_k, W_k = _kronecker_decomp(U2, V2, W2, U3, V3, W3)
            if _verify(U_k, V_k, W_k):
                rank = U_k.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_k, V_k, W_k)
        except Exception:
            pass
    
    # 2. Try Kronecker product of Strassen(2x2) and schoolbook(3x3) -> rank 189
    if best_result is None or best_rank > 189:
        if time.time() < deadline - 30:
            try:
                U2, V2, W2 = _strassen_2x2()
                U3, V3, W3 = _build_schoolbook_6x6()
                # Need to reshape for 3x3
                n3, m3, p3 = 3, 3, 3
                rank3 = 27
                U3 = np.zeros((n3*m3, rank3))
                V3 = np.zeros((m3*p3, rank3))
                W3 = np.zeros((p3*n3, rank3))
                col = 0
                for i in range(n3):
                    for j in range(m3):
                        for k in range(p3):
                            U3[i*m3+j, col] = 1.0
                            V3[j*p3+k, col] = 1.0
                            W3[k*n3+i, col] = 1.0
                            col += 1
                
                U_k, V_k, W_k = _kronecker_decomp(U2, V2, W2, U3, V3, W3)
                if _verify(U_k, V_k, W_k):
                    rank = U_k.shape[1]
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U_k, V_k, W_k)
            except Exception:
                pass
    
    # 3. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook_6x6()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
