"""Matrix multiplication 6,6,6 tensor decomposition using Kronecker products."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication, rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2
    
    # U[i*2+j, r]: coefficient of A[i,j] in product r
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # a00: M1, M3, M5
        [0, 0, 0, 0, 1, 0, 1],  # a01: M5, M7
        [0, 1, 0, 0, 0, 1, 0],  # a10: M2, M6
        [1, 1, 0, 1, 0, 0, -1], # a11: M1, M2, M4, -M7
    ]
    
    # V[j*2+k, r]: coefficient of B[j,k] in product r
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # b00: M1, M2, -M4, M6
        [0, 0, 1, 0, 0, 1, 0],   # b01: M3, M6
        [0, 0, 0, 1, 0, 0, 1],   # b10: M4, M7
        [1, 0, -1, 0, 1, 0, 1],  # b11: M1, -M3, M5, M7
    ]
    
    # W[k*2+i, r]: coefficient for C[k,i] from product r
    # C00 = M1 - M4 + M5 + M7 => W[0*2+0, r] = W[0, r]
    # C01 = M3 + M5 => W[0*2+1, r] = W[1, r]
    # C10 = M2 + M4 => W[1*2+0, r] = W[2, r]
    # C11 = M1 - M3 + M6 + M2 => W[1*2+1, r] = W[3, r]
    W = [
        [1, 0, 0, -1, 1, 0, 1],   # C00: M1, -M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],    # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0],    # C10: M2, M4
        [1, 1, -1, 0, 0, 1, 0],   # C11: M1, M2, -M3, M6
    ]
    
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """
    Build the schoolbook (trivial) decomposition of rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
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
    return U, V, W


def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combine two decompositions via Kronecker product.
    Given decomp1 for (n1,m1,p1) with rank r1 and decomp2 for (n2,m2,p2) with rank r2,
    produce a decomposition for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    The Kronecker product of two tensor decompositions gives a valid decomposition
    of the tensor product. For matrix multiplication, this corresponds to block 
    matrix multiplication where each block is itself a matrix product.
    
    Indexing convention:
    - For 6x6x6, we use 3x3x3 (schoolbook, rank 27) and 2x2x2 (Strassen, rank 7)
    - Result: 6x6x6 with rank 27*7 = 189
    
    The mapping:
    - A_6[i6, j6] where i6 = i3*n2 + i2, j6 = j3*m2 + j2
    - B_6[j6, k6] where j6 = j3*m2 + j2, k6 = k3*p2 + k2
    - C_6[i6, k6] where i6 = i3*n2 + i2, k6 = k3*p2 + k2
    
    U_6[i6*m6+j6, r] = U1[i3*m1+j3, r1] * U2[i2*m2+j2, r2]
    V_6[j6*p6+k6, r] = V1[j3*p1+k3, r1] * V2[j2*p2+k2, r2]
    W_6[k6*n6+i6, r] = W1[k3*n1+i3, r1] * W2[k2*n2+i2, r2]
    where r = r1*r2 + r2 (or some bijection from [0,r1*r2) to [0,r1)x[0,r2))
    """
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i3 in range(n1):
        for i2 in range(n2):
            i6 = i3 * n2 + i2
            for j3 in range(m1):
                for j2 in range(m2):
                    j6 = j3 * m2 + j2
                    a_idx = i6 * m + j6
                    for r1_idx in range(r1):
                        u1_val = U1[i3 * m1 + j3][r1_idx]
                        if u1_val == 0:
                            continue
                        for r2_idx in range(r2):
                            u2_val = U2[i2 * m2 + j2][r2_idx]
                            if u2_val == 0:
                                continue
                            r_idx = r1_idx * r2 + r2_idx
                            U[a_idx][r_idx] = u1_val * u2_val
    
    for j3 in range(m1):
        for j2 in range(m2):
            j6 = j3 * m2 + j2
            for k3 in range(p1):
                for k2 in range(p2):
                    k6 = k3 * p2 + k2
                    b_idx = j6 * p + k6
                    for r1_idx in range(r1):
                        v1_val = V1[j3 * p1 + k3][r1_idx]
                        if v1_val == 0:
                            continue
                        for r2_idx in range(r2):
                            v2_val = V2[j2 * p2 + k2][r2_idx]
                            if v2_val == 0:
                                continue
                            r_idx = r1_idx * r2 + r2_idx
                            V[b_idx][r_idx] = v1_val * v2_val
    
    for k3 in range(p1):
        for k2 in range(p2):
            k6 = k3 * p2 + k2
            for i3 in range(n1):
                for i2 in range(n2):
                    i6 = i3 * n2 + i2
                    c_idx = k6 * n + i6
                    for r1_idx in range(r1):
                        w1_val = W1[k3 * n1 + i3][r1_idx]
                        if w1_val == 0:
                            continue
                        for r2_idx in range(r2):
                            w2_val = W2[k2 * n2 + i2][r2_idx]
                            if w2_val == 0:
                                continue
                            r_idx = r1_idx * r2 + r2_idx
                            W[c_idx][r_idx] = w1_val * w2_val
    
    return U, V, W


def _validate(U, V, W):
    """
    Validate that U, V, W form an exact tensor decomposition of matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r] should equal 1 iff consistent.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[0] != n * m:
        return False
    if V_arr.shape[0] != m * p:
        return False
    if W_arr.shape[0] != p * n:
        return False
    if U_arr.shape[1] != V_arr.shape[1] or V_arr.shape[1] != W_arr.shape[1]:
        return False
    
    rank = U_arr.shape[1]
    
    # Compute decomposition tensor D
    D = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    if not np.allclose(D, T, atol=1e-9):
        return False
    
    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 6x6x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Use Kronecker product of 3x3x3 schoolbook (rank 27) and 2x2x2 Strassen (rank 7)
    # to get rank 189 for 6x6x6.
    
    U3, V3, W3 = _schoolbook_nm_p(3, 3, 3)
    U2, V2, W2 = _strassen_2x2x2()
    
    U, V, W = _kron_decomp(U3, V3, W3, U2, V2, W2, 3, 3, 3, 2, 2, 2)
    
    # Validate
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook 6x6x6 (Rank 216)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
