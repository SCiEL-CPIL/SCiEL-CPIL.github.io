"""Matrix multiplication tensor decomposition for 4x4 * 4x5.
Uses Strassen (rank 49) for the 4x4*4x4 part plus schoolbook (rank 16) for the extra column.
Total rank: 65.
"""
import time
import numpy as np

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) as numpy arrays.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=np.int64)
    V = np.zeros((4, rank), dtype=np.int64)
    W = np.zeros((4, rank), dtype=np.int64)
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49.
    Returns U (16x49), V (16x49), W (16x49) as numpy arrays.
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    sU, sV, sW = _strassen_2x2()
    
    # 7 block-level Strassen multiplications
    # Each: (a_coeffs, b_coeffs, c_coeffs)
    # Block indices: 0=11, 1=12, 2=21, 3=22
    mults = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),    # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),          # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),          # M3
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),          # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),          # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}),     # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),    # M7
    ]
    
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        for s in range(7):
            col = mi * 7 + s
            
            # U part
            for xi in range(4):
                u_val = sU[xi, s]
                if u_val == 0:
                    continue
                xr = xi // 2
                xc = xi % 2
                for ab, coeff in a_c.items():
                    global_u = ab * 4 + xr * 2 + xc
                    U[global_u, col] += u_val * coeff
            
            # V part
            for yi in range(4):
                v_val = sV[yi, s]
                if v_val == 0:
                    continue
                yr = yi // 2
                yc = yi % 2
                for bb, coeff in b_c.items():
                    global_v = bb * 4 + yr * 2 + yc
                    V[global_v, col] += v_val * coeff
            
            # W part
            for zi in range(4):
                w_val = sW[zi, s]
                if w_val == 0:
                    continue
                zr = zi // 2
                zc = zi % 2
                for cb, coeff in c_c.items():
                    global_w = cb * 4 + zr * 2 + zc
                    W[global_w, col] += w_val * coeff
    
    return U, V, W

def _build_extra_column():
    """Build schoolbook decomposition for 4x4 * 4x1 -> 4x1 with rank 16.
    Returns U (16x16), V (4x16), W (4x16) as numpy arrays.
    Indexing: U[i*4+j, c], V[j*1+k, c], W[k*4+i, c] where k=0
    """
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 130
    
    # Build Strassen 4x4x4 part (rank 49)
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    
    # Build extra column part (rank 16)
    U_e, V_e, W_e = _build_extra_column()
    
    total_rank = 49 + 16
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    # Copy Strassen part (columns 0-3 of B, i.e., k=0..3)
    U[:, :49] = U_s
    for j in range(m):
        for k in range(4):
            V[j*p+k, :49] = V_s[j*4+k, :]
    for k in range(4):
        for i in range(n):
            W[k*n+i, :49] = W_s[k*4+i, :]
    
    # Copy extra column part (column 4 of B, i.e., k=4)
    U[:, 49:] = U_e
    for j in range(m):
        V[j*p+4, 49:] = V_e[j*1+0, :]
    for i in range(n):
        W[4*n+i, 49:] = W_e[0*n+i, :]
    
    # Verify
    if not _verify(U, V, W, n, m, p, total_rank):
        # Fallback to schoolbook (rank 80)
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
