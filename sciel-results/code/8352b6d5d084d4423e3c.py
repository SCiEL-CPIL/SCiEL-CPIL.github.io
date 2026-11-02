import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4,7) indexed by (i,j) -> i*2+j
    V: (4,7) indexed by (j,k) -> j*2+k
    W: (4,7) indexed by (k,i) -> k*2+i
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 1, 0, -1],   # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],    # B00
        [0, 0, 0, 1, 0, 0, 1],    # B01
        [1, 0, -1, -1, 0, 1, 0],  # B10
        [1, 0, 0, 0, 1, 0, 1]     # B11
    ], dtype=float)
    
    # W indexed by (k,i): index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C[0,0]=M1-M4+M5-M7, C[1,0]=M2+M3, C[0,1]=M3+M4, C[1,1]=M1-M2+M6-M7
    W = np.array([
        [1, 0, 0, -1, 1, 0, -1],  # C[0,0], k=0,i=0
        [0, 1, 1, 0, 0, 0, 0],    # C[1,0], k=0,i=1
        [0, 0, 1, 1, 0, 0, 0],    # C[0,1], k=1,i=0
        [1, -1, 0, 0, 0, 1, -1]   # C[1,1], k=1,i=1
    ], dtype=float)
    
    return U, V, W

def _two_by_two_by_four():
    """
    Returns U, V, W for 2x2x4 matrix multiplication (rank 14 via two Strassen).
    U: (4, 14) indexed by (i,j) -> i*2+j
    V: (8, 14) indexed by (j,k) -> j*4+k
    W: (8, 14) indexed by (k,i) -> k*2+i
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank = 14
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    # First half: B columns 0,1
    U[:, 0:7] = U_s
    V[0:4, 0:7] = V_s
    # W: C[k,i] for k=0,1 -> indices k*2+i
    W[0:4, 0:7] = W_s
    
    # Second half: B columns 2,3
    U[:, 7:14] = U_s
    V[4:8, 7:14] = V_s
    # W: C[k,i] for k=2,3 -> indices k*2+i
    W[4:8, 7:14] = W_s
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(np.dot(U_np[a, :], V_np[b, :]), W_np[c, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_kronecker():
    """
    Construct 4x4x8 decomposition via Kronecker product of 2x2x2 and 2x2x4.
    Total rank = 7 * 14 = 98.
    """
    n, m, p = 4, 4, 8
    
    U_s, V_s, W_s = _strassen_2x2x2()
    U_224, V_224, W_224 = _two_by_two_by_four()
    
    rank_s = U_s.shape[1]   # 7
    rank_224 = U_224.shape[1]  # 14
    total_rank = rank_s * rank_224  # 98
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for r_s in range(rank_s):
        for r_224 in range(rank_224):
            r = r_s * rank_224 + r_224
            
            # U: (i,j) = (i_hi*2+i_lo, j_hi*2+j_lo)
            # U index = i*4+j = i_hi*8+i_lo*4+j_hi*2+j_lo
            for i_hi in range(2):
                for j_hi in range(2):
                    cs = U_s[i_hi * 2 + j_hi, r_s]
                    if cs == 0:
                        continue
                    for i_lo in range(2):
                        for j_lo in range(2):
                            c224 = U_224[i_lo * 2 + j_lo, r_224]
                            if c224 == 0:
                                continue
                            u_idx = i_hi * 8 + i_lo * 4 + j_hi * 2 + j_lo
                            U[u_idx, r] = cs * c224
            
            # V: (j,k) = (j_hi*2+j_lo, k_hi*4+k_lo)
            # V index = j*8+k = j_hi*16+j_lo*8+k_hi*4+k_lo
            for j_hi in range(2):
                for k_hi in range(2):
                    cs = V_s[j_hi * 2 + k_hi, r_s]
                    if cs == 0:
                        continue
                    for j_lo in range(2):
                        for k_lo in range(4):
                            c224 = V_224[j_lo * 4 + k_lo, r_224]
                            if c224 == 0:
                                continue
                            v_idx = j_hi * 16 + j_lo * 8 + k_hi * 4 + k_lo
                            V[v_idx, r] = cs * c224
            
            # W: (k,i) = (k_hi*4+k_lo, i_hi*2+i_lo)
            # W index = k*4+i = k_hi*16+k_lo*4+i_hi*2+i_lo
            for k_hi in range(2):
                for i_hi in range(2):
                    cs = W_s[k_hi * 2 + i_hi, r_s]
                    if cs == 0:
                        continue
                    for k_lo in range(4):
                        for i_lo in range(2):
                            c224 = W_224[k_lo * 2 + i_lo, r_224]
                            if c224 == 0:
                                continue
                            w_idx = k_hi * 16 + k_lo * 4 + i_hi * 2 + i_lo
                            W[w_idx, r] = cs * c224
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        U, V, W = _construct_kronecker()
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
    
    # Fallback: schoolbook
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
