import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1
    # A: A11(0,0)=1, A22(1,1)=1
    # B: B11(0,0)=1, B22(1,1)=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C: C11(0,0)=1, C22(1,1)=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2
    # A: A21(1,0)=1, A22(1,1)=1
    # B: B11(0,0)=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C: C21(1,0)=1
    W[2, 1] = 1
    
    # Product 3
    # A: A11(0,0)=1
    # B: B21(1,0)=1, B22(1,1)=-1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C: C12(0,1)=1, C22(1,1)=1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 4
    # A: A22(1,1)=1
    # B: B12(0,1)=1, B11(0,0)=-1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C: C11(0,0)=1, C21(1,0)=1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 5
    # A: A11(0,0)=1, A12(0,1)=1
    # B: B22(1,1)=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C: C11(0,0)=-1, C12(0,1)=1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 6
    # A: A21(1,0)=1, A11(0,0)=-1
    # B: B11(0,0)=1, B12(0,1)=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C: C22(1,1)=1
    W[3, 5] = 1
    
    # Product 7
    # A: A12(0,1)=1, A22(1,1)=-1
    # B: B21(1,0)=1, B22(1,1)=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C: C11(0,0)=1
    W[0, 6] = 1
    
    return U, V, W

def get_schoolbook_2x2x1():
    """
    Returns U, V, W for 2x2x1 multiplication with rank 4.
    U shape (4, 4), V shape (2, 4), W shape (2, 4).
    Note: V shape is m*p = 2*1 = 2. W shape is p*n = 1*2 = 2.
    Wait, indices:
    U: i*2+j (4 entries)
    V: j*1+k (2 entries, k=0)
    W: k*2+i (2 entries, k=0)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Term 1: A00 * B0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 2: A01 * B1
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 3: A10 * B0
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 4: A11 * B1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def kron_tensor(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Combines two tensor decompositions using Kronecker product.
    T1: n1 x m1 x p1, rank r1
    T2: n2 x m2 x p2, rank r2
    Result: n1*n2 x m1*m2 x p1*p2, rank r1*r2
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Indices for T1: i1, j1, k1
    # Indices for T2: i2, j2, k2
    # Global indices: i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    # U index: i*m + j = (i1*n2 + i2)*m + j = (i1*n2 + i2)*(m1*m2) + j1*m2 + j2
    #         = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
    # V index: j*p + k = (j1*m2 + j2)*(p1*p2) + k = j1*m2*p1*p2 + j2*p1*p2 + k1*p2 + k2
    # W index: k*n + i = (k1*p2 + k2)*(n1*n2) + i = k1*p2*n1*n2 + k2*n1*n2 + i1*n2 + i2
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            
            # Get non-zero elements? No, just iterate all.
            # It's small, so direct loops are fine.
            for i1 in range(n1):
                for j1 in range(m1):
                    for k1 in range(p1):
                        u1_val = U1[i1 * m1 + j1, r1_idx]
                        v1_val = V1[j1 * p1 + k1, r1_idx]
                        w1_val = W1[k1 * n1 + i1, r1_idx]
                        
                        if u1_val == 0 and v1_val == 0 and w1_val == 0:
                            continue
                            
                        for i2 in range(n2):
                            for j2 in range(m2):
                                for k2 in range(p2):
                                    u2_val = U2[i2 * m2 + j2, r2_idx]
                                    v2_val = V2[j2 * p2 + k2, r2_idx]
                                    w2_val = W2[k2 * n2 + i2, r2_idx]
                                    
                                    if u2_val == 0 and v2_val == 0 and w2_val == 0:
                                        continue
                                    
                                    u_val = u1_val * u2_val
                                    v_val = v1_val * v2_val
                                    w_val = w1_val * w2_val
                                    
                                    i = i1 * n2 + i2
                                    j = j1 * m2 + j2
                                    k = k1 * p2 + k2
                                    
                                    u_idx = i * m + j
                                    v_idx = j * p + k
                                    w_idx = k * n + i
                                    
                                    U[u_idx, r_idx] = u_val
                                    V[v_idx, r_idx] = v_val
                                    W[w_idx, r_idx] = w_val
                                    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 (Rank 49)
    # T444 = T222 (Strassen) x T222 (Strassen)
    U_s, V_s, W_s = get_strassen_2x2x2()
    U_444, V_444, W_444 = kron_tensor(U_s, V_s, W_s, U_s, V_s, W_s, 2, 2, 2, 2, 2, 2)
    
    if time.time() > deadline:
        # Fallback to schoolbook if too slow (unlikely)
        return _schoolbook_construct()
        
    # Part 2: 4x4x2 (Rank 28)
    # T442 = T222 (Strassen) x T221 (Schoolbook)
    U_t, V_t, W_t = get_schoolbook_2x2x1()
    U_442, V_442, W_442 = kron_tensor(U_s, V_s, W_s, U_t, V_t, W_t, 2, 2, 2, 2, 2, 1)
    
    if time.time() > deadline:
        return _schoolbook_construct()
        
    # Combine
    # U_444 shape (16, 49), V_444 shape (16, 49), W_444 shape (16, 49)
    # U_442 shape (16, 28), V_442 shape (8, 28), W_442 shape (8, 28)
    
    # Wait, V and W shapes for 4x4x2:
    # m*p = 4*2 = 8.
    # p*n = 2*4 = 8.
    # So V_442 is (8, 28), W_442 is (8, 28).
    
    # We need to align W_442 to columns 4 and 5 of the full 4x4x6 output.
    # Full W shape is (p*n, rank) = (6*4, rank) = (24, rank).
    # W_444 uses k=0..3. Indices k*4+i.
    # W_442 uses k=0..1 locally. We need to map to k=4..5.
    
    W_full = np.zeros((24, 77))
    # Copy W_444
    W_full[:16, :49] = W_444
    
    # Shift W_442
    # W_442 index is k_loc*4 + i.
    # We want k_glob = k_loc + 4.
    # So index is (k_loc+4)*4 + i = k_loc*4 + 16 + i.
    W_full[16:24, 49:] = W_442
    
    U_full = np.hstack([U_444, U_442])
    V_full = np.hstack([V_444, V_442])
    
    # Convert to lists
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }

def _schoolbook_construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
