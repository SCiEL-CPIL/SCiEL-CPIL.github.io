import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Returns list of 7 tuples (u, v, w) for 2x2 matrix multiplication.
    u, v, w are lists of 4 coefficients corresponding to flattened indices 0..3.
    Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    t1_u = [1, 0, 0, 1]
    t1_v = [1, 0, 0, 1]
    t1_w = [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    t2_u = [0, 0, 1, 1]
    t2_v = [1, 0, 0, 0]
    t2_w = [0, 0, 1, -1]
    
    # M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    t3_u = [1, 0, 0, 0]
    t3_v = [0, 1, 0, -1]
    t3_w = [0, 1, 0, 1]
    
    # M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    t4_u = [0, 0, 0, 1]
    t4_v = [0, 0, 1, -1] # B10 is idx 2, B00 is idx 0. So [0,0,1,-1]
    t4_w = [1, 0, 1, 0]
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    t5_u = [1, 1, 0, 0]
    t5_v = [0, 0, 0, 1]
    t5_w = [-1, 1, 0, 0]
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    t6_u = [0, 0, 1, -1] # A10 is idx 2, A00 is idx 0. [0,0,1,-1]
    t6_v = [1, 1, 0, 0]
    t6_w = [0, 0, 0, 1]
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    t7_u = [0, 1, 0, -1] # A01 is idx 1, A11 is idx 3. [0,1,0,-1]
    t7_v = [0, 0, 1, 1]
    t7_w = [1, 0, 0, 0]
    
    return [
        (t1_u, t1_v, t1_w),
        (t2_u, t2_v, t2_w),
        (t3_u, t3_v, t3_w),
        (t4_u, t4_v, t4_w),
        (t5_u, t5_v, t5_w),
        (t6_u, t6_v, t6_w),
        (t7_u, t7_v, t7_w)
    ]

def _construct_4x4_terms():
    """
    Construct the 49 rank-1 terms for 4x4 matrix multiplication using Strassen.
    Returns list of 49 tuples (u, v, w) where u,v,w are lists of 16 coefficients.
    """
    s2_terms = _strassen_2x2_terms()
    terms_4x4 = []
    
    for r in range(7):
        u1, v1, w1 = s2_terms[r]
        for s in range(7):
            u0, v0, w0 = s2_terms[s]
            
            u4 = [0] * 16
            v4 = [0] * 16
            w4 = [0] * 16
            
            # Map 4x4 indices i,j to block indices i1,j1 and intra-block indices i0,j0
            # i = i1*2 + i0, j = j1*2 + j0
            # Flattened index = i*4 + j = (i1*2+i0)*4 + (j1*2+j0) = i1*8 + i0*4 + j1*2 + j0
            
            for i1 in range(2):
                for j1 in range(2):
                    coeff_block = u1[i1*2 + j1]
                    if coeff_block == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            coeff_inner = u0[i0*2 + j0]
                            if coeff_inner == 0: continue
                            coeff_total = coeff_block * coeff_inner
                            idx = (i1*2 + i0)*4 + (j1*2 + j0)
                            u4[idx] = coeff_total
                            
            for i1 in range(2):
                for j1 in range(2):
                    coeff_block = v1[i1*2 + j1]
                    if coeff_block == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            coeff_inner = v0[i0*2 + j0]
                            if coeff_inner == 0: continue
                            coeff_total = coeff_block * coeff_inner
                            idx = (i1*2 + i0)*4 + (j1*2 + j0)
                            v4[idx] = coeff_total
                            
            for i1 in range(2):
                for j1 in range(2):
                    coeff_block = w1[i1*2 + j1]
                    if coeff_block == 0: continue
                    for i0 in range(2):
                        for j0 in range(2):
                            coeff_inner = w0[i0*2 + j0]
                            if coeff_inner == 0: continue
                            coeff_total = coeff_block * coeff_inner
                            idx = (i1*2 + i0)*4 + (j1*2 + j0)
                            w4[idx] = coeff_total
                            
            terms_4x4.append((u4, v4, w4))
            
    return terms_4x4

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p):
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    # Verify for all u, v, w
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                i_u = u // m
                j_u = u % m
                j_v = v // p
                k_v = v % p
                k_w = w // n
                i_w = w % n
                
                expected = 1.0 if (j_u == j_v and k_v == k_w and i_w == i_u) else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U[u, r] * V[v, r] * W[w, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_5x5():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Add 4x4 core terms (indices 0..3)
    core_terms = _construct_4x4_terms()
    for r_idx in range(49):
        u4, v4, w4 = core_terms[r_idx]
        for i in range(4):
            for j in range(4):
                u_idx_4 = i * 4 + j
                u_idx_5 = i * 5 + j
                U[u_idx_5][r_idx] = u4[u_idx_4]
                
                v_idx_4 = i * 4 + j
                v_idx_5 = i * 5 + j
                V[v_idx_5][r_idx] = v4[v_idx_4]
                
                w_idx_4 = i * 4 + j
                w_idx_5 = i * 5 + j
                W[w_idx_5][r_idx] = w4[w_idx_4]

    # 2. Add boundary terms (involving index 4)
    r_offset = 49
    
    # Type 1: i in 0..3, j=4, k in 0..3 -> C[i,k]
    # 1 term
    r = r_offset
    for i in range(4):
        U[i * 5 + 4][r] = 1
    for k in range(4):
        V[4 * 5 + k][r] = 1
    for i in range(4):
        for k in range(4):
            W[k * 5 + i][r] = 1
    r += 1
    
    # Type 2: i in 0..3, j in 0..3, k=4 -> C[i,4]
    # 1 term
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][r] = 1
    for j in range(4):
        V[j * 5 + 4][r] = 1
    for i in range(4):
        W[4 * 5 + i][r] = 1
    r += 1
    
    # Type 3: i=4, j in 0..3, k in 0..3 -> C[4,k]
    # 1 term
    for j in range(4):
        U[4 * 5 + j][r] = 1
    for j in range(4):
        for k in range(4):
            V[j * 5 + k][r] = 1
    for k in range(4):
        W[k * 5 + 4][r] = 1
    r += 1
    
    # Type 4: i=4, j=4, k in 0..3 -> C[4,k]
    # 1 term
    U[4 * 5 + 4][r] = 1
    for k in range(4):
        V[4 * 5 + k][r] = 1
    for k in range(4):
        W[k * 5 + 4][r] = 1
    r += 1
    
    # Type 5: i=4, j in 0..3, k=4 -> C[4,4]
    # 1 term
    for j in range(4):
        U[4 * 5 + j][r] = 1
    for j in range(4):
        V[j * 5 + 4][r] = 1
    W[4 * 5 + 4][r] = 1
    r += 1
    
    # Type 6: i in 0..3, j=4, k=4 -> C[i,4]
    # 1 term
    for i in range(4):
        U[i * 5 + 4][r] = 1
    V[4 * 5 + 4][r] = 1
    for i in range(4):
        W[4 * 5 + i][r] = 1
    r += 1
    
    # Type 7: i=4, j=4, k=4 -> C[4,4]
    # 1 term
    U[4 * 5 + 4][r] = 1
    V[4 * 5 + 4][r] = 1
    W[4 * 5 + 4][r] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    decomp = _construct_5x5()
    
    # Verify to be safe
    if _verify(decomp, n, m, p):
        return decomp
    
    # Fallback to schoolbook if verification fails
    return _schoolbook()
# EVOLVE-BLOCK-END
