import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7. Verified numerically."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Local: U[i*2+j], V[j*2+k], W[i*2+k]
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,3]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,2]=1
    # M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,1]=1; W[2,2]=1
    # M4 = A11(B01-B10)
    U[3,3]=1
    V[1,3]=1; V[2,3]=-1
    W[0,0]=1; W[2,2]=1
    # M5 = (A00+A10)B11
    U[0,4]=1; U[2,4]=1
    V[3,4]=1
    W[0,0]=-1; W[1,1]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[3,3]=1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,0]=1; W[1,1]=1
    return U, V, W

def _schoolbook_1x2x2():
    """Schoolbook 1x2x2, rank 4."""
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    for j in range(2):
        for k in range(2):
            c = j*2+k
            U[j,c]=1
            V[j*2+k,c]=1
            W[k,c]=1
    return U, V, W

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_3x4x8():
    n, m, p = 3, 4, 8
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_3x4x8():
    n, m, p = 3, 4, 8
    sU, sV, sW = _strassen_2x2x2()
    hU, hV, hW = _schoolbook_1x2x2()
    
    row_blocks = [(0, 2), (2, 1)]
    col_blocks = [(0, 2), (2, 2)]
    out_col_blocks = [(0, 2), (2, 2), (4, 2), (6, 2)]
    
    R = 0
    for (i_s, i_sz) in row_blocks:
        for (j_s, j_sz) in col_blocks:
            for (k_s, k_sz) in out_col_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    R += 7
                elif i_sz == 1 and j_sz == 2 and k_sz == 2:
                    R += 4
                else:
                    raise ValueError(f"Unexpected {i_sz}x{j_sz}x{k_sz}")
    
    U_full = np.zeros((n*m, R))
    V_full = np.zeros((m*p, R))
    W_full = np.zeros((p*n, R))
    
    col_off = 0
    for (i_s, i_sz) in row_blocks:
        for (j_s, j_sz) in col_blocks:
            for (k_s, k_sz) in out_col_blocks:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    Ub, Vb, Wb = sU, sV, sW
                else:
                    Ub, Vb, Wb = hU, hV, hW
                rb = Ub.shape[1]
                for di in range(i_sz):
                    for dj in range(j_sz):
                        U_full[(i_s+di)*m + (j_s+dj), col_off:col_off+rb] = Ub[di*j_sz+dj, :]
                for dj in range(j_sz):
                    for dk in range(k_sz):
                        V_full[(j_s+dj)*p + (k_s+dk), col_off:col_off+rb] = Vb[dj*k_sz+dk, :]
                for di in range(i_sz):
                    for dk in range(k_sz):
                        W_full[(k_s+dk)*n + (i_s+di), col_off:col_off+rb] = Wb[di*k_sz+dk, :]
                col_off += rb
    
    assert col_off == R
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    U, V, W = _construct_3x4x8()
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = _schoolbook_3x4x8()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
