import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _make_schoolbook(n, m, p):
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
    return U, V, W

def _make_strassen_2x2x2():
    # n=2, m=2, p=2
    # Returns U (4, 7), V (4, 7), W (4, 7)
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (W index is k*n+i, so 0*2+0=0, 1*2+0=2, 0*2+1=1, 1*2+1=3)
    # Wait, W index is k*n+i.
    # C_00 -> k=0, i=0 -> 0
    # C_01 -> k=1, i=0 -> 2
    # C_10 -> k=0, i=1 -> 1
    # C_11 -> k=1, i=1 -> 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A: 00(1), 11(1) -> U[0]=1, U[3]=1
    # B: 00(1), 11(1) -> V[0]=1, V[3]=1
    # C: C00(1), C11(1) -> W[0]=1, W[3]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # A: 10(1), 11(1) -> U[2]=1, U[3]=1
    # B: 00(1) -> V[0]=1
    # C: C10(1), C11(-1) -> W[1]=1, W[3]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    # A: 00(1) -> U[0]=1
    # B: 01(1), 11(-1) -> V[1]=1, V[3]=-1
    # C: C01(1), C11(1) -> W[2]=1, W[3]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    # A: 11(1) -> U[3]=1
    # B: 10(1), 00(-1) -> V[2]=1, V[0]=-1
    # C: C10(1), C00(1) -> W[1]=1, W[0]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = 1
    
    # M5 = (A00+A01)B11
    # A: 00(1), 01(1) -> U[0]=1, U[1]=1
    # B: 11(1) -> V[3]=1
    # C: C11(-1), C01(1) -> W[3]=-1, W[2]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[3, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # A: 10(1), 00(-1) -> U[2]=1, U[0]=-1
    # B: 00(1), 01(1) -> V[0]=1, V[1]=1
    # C: C11(1) -> W[3]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A: 01(1), 11(-1) -> U[1]=1, U[3]=-1
    # B: 10(1), 11(1) -> V[2]=1, V[3]=1
    # C: C00(1) -> W[0]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed(U_loc, V_loc, W_loc, n_g, m_g, p_g, i_off, j_off, k_off, U_g, V_g, W_g, col_off):
    n_l, m_l, p_l = U_loc.shape[0]//m_l, m_l, V_loc.shape[0]//p_l
    # Actually, we know n_l, m_l, p_l from context
    # Let's pass them explicitly or infer
    # U_loc is (n_l * m_l, R)
    # V_loc is (m_l * p_l, R)
    # W_loc is (p_l * n_l, R)
    
    R = U_loc.shape[1]
    for r in range(R):
        for i in range(n_l):
            for j in range(m_l):
                val = U_loc[i * m_l + j, r]
                if val != 0:
                    I = i_off + i
                    J = j_off + j
                    U_g[I * m_g + J, col_off + r] += val
                    
        for j in range(m_l):
            for k in range(p_l):
                val = V_loc[j * p_l + k, r]
                if val != 0:
                    J = j_off + j
                    K = k_off + k
                    V_g[J * p_g + K, col_off + r] += val
                    
        for k in range(p_l):
            for i in range(n_l):
                val = W_loc[k * n_l + i, r]
                if val != 0:
                    K = k_off + k
                    I = i_off + i
                    W_g[K * n_g + I, col_off + r] += val

def construct():
    n, m, p = DIMENSIONS
    # Total rank 68
    rank = 68
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 2: 3x1x8 (offset i=0, j=2, k=0)
    U2, V2, W2 = _make_schoolbook(3, 1, 8)
    _embed(U2, V2, W2, n, m, p, 0, 2, 0, U, V, W, col)
    col += U2.shape[1] # 24
    
    # Part 1: 3x2x8 (offset i=0, j=0, k=0)
    # Sub-part 1b: 1x2x8 (offset i=2, j=0, k=0)
    U1b, V1b, W1b = _make_schoolbook(1, 2, 8)
    _embed(U1b, V1b, W1b, n, m, p, 2, 0, 0, U, V, W, col)
    col += U1b.shape[1] # 16
    
    # Sub-part 1a: 2x2x8 (offset i=0, j=0, k=0)
    # Sub-sub-part 1a2: 2x2x4 (offset i=0, j=0, k=4)
    # Sub-sub-sub-part 1a22: 2x2x2 (offset i=0, j=0, k=6)
    U1a22, V1a22, W1a22 = _make_strassen_2x2x2()
    _embed(U1a22, V1a22, W1a22, n, m, p, 0, 0, 6, U, V, W, col)
    col += U1a22.shape[1] # 7
    
    # Sub-sub-sub-part 1a21: 2x2x2 (offset i=0, j=0, k=4)
    U1a21, V1a21, W1a21 = _make_strassen_2x2x2()
    _embed(U1a21, V1a21, W1a21, n, m, p, 0, 0, 4, U, V, W, col)
    col += U1a21.shape[1] # 7
    
    # Sub-sub-part 1a1: 2x2x4 (offset i=0, j=0, k=0)
    # Sub-sub-sub-part 1a12: 2x2x2 (offset i=0, j=0, k=2)
    U1a12, V1a12, W1a12 = _make_strassen_2x2x2()
    _embed(U1a12, V1a12, W1a12, n, m, p, 0, 0, 2, U, V, W, col)
    col += U1a12.shape[1] # 7
    
    # Sub-sub-sub-part 1a11: 2x2x2 (offset i=0, j=0, k=0)
    U1a11, V1a11, W1a11 = _make_strassen_2x2x2()
    _embed(U1a11, V1a11, W1a11, n, m, p, 0, 0, 0, U, V, W, col)
    col += U1a11.shape[1] # 7
    
    return {"U": U, "V": V, "W": W}
