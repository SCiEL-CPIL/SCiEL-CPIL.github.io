"""Matrix multiplication 2x4 x 4x6 via Kronecker product: Strassen(2x2x2) ⊗ schoolbook(1x2x3)."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2 multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U[i*2+j, c] coefficients for A[i,j] in product c
    # V[j*2+k, c] coefficients for B[j,k] in product c
    # W[k*2+i, c] coefficients for C[i,k] from product c
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # But W is indexed as W[k*2+i], so:
    # W[0*2+0]=W[0]=C[0,0], W[1*2+0]=W[2]=C[0,1], W[0*2+1]=W[1]=C[1,0], W[1*2+1]=W[3]=C[1,1]
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1  # a21
    U[3, 1] = 1  # a22
    V[0, 1] = 1  # b11
    W[2, 1] = 1  # C[1,0]
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1  # a11
    V[1, 2] = 1  # b12
    V[3, 2] = -1  # -b22
    W[1, 2] = 1  # C[0,1]
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1  # a22
    V[2, 3] = 1  # b21
    V[0, 3] = -1  # -b11
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1  # a11
    U[1, 4] = 1  # a12
    V[3, 4] = 1  # b22
    W[1, 4] = 1  # C[0,1]
    W[0, 4] = -1  # -C[0,0]
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1  # a21
    U[0, 5] = -1  # -a11
    V[0, 5] = 1  # b11
    V[1, 5] = 1  # b12
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1  # a12
    U[3, 6] = -1  # -a22
    V[2, 6] = 1  # b21
    V[3, 6] = 1  # b22
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _schoolbook_1x2x3():
    """Return schoolbook rank-6 decomposition for 1x2 x 2x3 -> 1x3 multiplication.
    This is just a row vector times a matrix.
    U: (2, 6), V: (6, 6), W: (3, 6)
    Indexing: U[j2] for the 1x2 vector (j2=0,1)
    V[j2*3+k2] for the 2x3 matrix
    W[k2] for the 1x3 result
    """
    # A is 1x2: A[0,j2], j2=0,1
    # B is 2x3: B[j2,k2], j2=0,1, k2=0,1,2
    # C is 1x3: C[0,k2], k2=0,1,2
    # C[0,k2] = sum_{j2} A[0,j2] * B[j2,k2]
    
    U = np.zeros((2, 6))
    V = np.zeros((6, 6))
    W = np.zeros((3, 6))
    
    for j2 in range(2):
        for k2 in range(3):
            c = j2 * 3 + k2
            U[j2, c] = 1
            V[j2 * 3 + k2, c] = 1
            W[k2, c] = 1
    
    return U, V, W


def _verify(U, V, W, n=2, m=4, p=6, tol=1e-10):
    """Verify that sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c] = δ_{i,i'}*δ_{j,j'}*δ_{k,k'}"""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            if abs(val - expected) > tol:
                                return abs(val - expected)
    return 0.0


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Kronecker product approach:
    # 2x4x6 = (2x2x2) ⊗ (1x2x3)
    # i: 2 (just i)
    # j: 4 = 2(j1) x 2(j2), j = j1*2 + j2
    # k: 6 = 2(k1) x 3(k2), k = k1*3 + k2
    
    # Outer: 2x2x2 multiplication (i, j1, k1) - rank 7 (Strassen)
    # Inner: 1x2x3 multiplication (j2, k2) - rank 6 (schoolbook)
    # Total rank: 7 * 6 = 42
    
    U_outer, V_outer, W_outer = _strassen_2x2x2()
    U_inner, V_inner, W_inner = _schoolbook_1x2x3()
    
    r_out = U_outer.shape[1]  # 7
    r_in = U_inner.shape[1]   # 6
    r_total = r_out * r_in    # 42
    
    U = np.zeros((n * m, r_total))  # (8, 42)
    V = np.zeros((m * p, r_total))  # (24, 42)
    W = np.zeros((p * n, r_total))  # (12, 42)
    
    # j = j1*2 + j2, where j1 ∈ {0,1}, j2 ∈ {0,1}
    # k = k1*3 + k2, where k1 ∈ {0,1,2}, k2 ∈ {0,1,2}
    # i = i (just 2)
    
    # U[i*m+j, c] = U[i*4 + j1*2+j2, c_out*6+c_in]
    #   = U_outer[i*2+j1, c_out] * U_inner[j2, c_in]
    
    # V[j'*p+k, c] = V[j'*6 + k1*3+k2, c_out*6+c_in]
    #   = V_outer[j1'*2+k1, c_out] * V_inner[j2'*3+k2, c_in]
    # Wait, V is indexed by (j', k) where j' is the row of B (size 4) and k is the col of B (size 6)
    # j' = j1'*2 + j2', k = k1*3 + k2
    # V[j'*p+k, c] = V[j1'*2+j2', k1*3+k2, c]
    # = V_outer[j1'*2+k1, c_out] * V_inner[j2'*3+k2, c_in]
    
    # W[k'*n+i', c] = W[k'*2+i', c_out*6+c_in]
    # k' = k1'*3 + k2', i' = i'
    # W[k'*n+i', c] = W[k1'*3+k2', i', c]
    # = W_outer[k1'*2+i', c_out] * W_inner[k2', c_in]
    
    for c_out in range(r_out):
        for c_in in range(r_in):
            c = c_out * r_in + c_in
            
            # Fill U
            for i in range(n):  # i = 0,1
                for j1 in range(2):
                    for j2 in range(2):
                        j = j1 * 2 + j2
                        U[i * m + j, c] = U_outer[i * 2 + j1, c_out] * U_inner[j2, c_in]
            
            # Fill V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(3):
                        for k2 in range(3):
                            jp = j1 * 2 + j2
                            k = k1 * 3 + k2
                            V[jp * p + k, c] = V_outer[j1 * 2 + k1, c_out] * V_inner[j2 * 3 + k2, c_in]
            
            # Fill W
            for k1 in range(3):
                for k2 in range(3):
                    for ip in range(n):
                        kp = k1 * 3 + k2
                        W[kp * n + ip, c] = W_outer[k1 * 2 + ip, c_out] * W_inner[k2, c_in]
    
    # Verify
    err = _verify(U, V, W, n, m, p, tol=1e-10)
    if err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
