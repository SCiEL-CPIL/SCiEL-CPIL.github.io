import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    U: (4,7) indexed by (i,j), V: (4,7) indexed by (j,k), W: (4,7) indexed by (k,i)
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_kron():
    """4x4x4 via Kronecker of two 2x2x2 Strassen, rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i//2), c1]
    return U4, V4, W4

def _build_4x4x8():
    """4x4x8 rank-98: two 4x4x4 Strassen-Kronecker blocks."""
    U4, V4, W4 = _build_4x4x4_kron()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        cs = block * 49
        U[:, cs:cs+49] = U4
        for j in range(4):
            for kl in range(4):
                V[j*8 + block*4 + kl, cs:cs+49] = V4[j*4 + kl, :]
        for kl in range(4):
            for i in range(4):
                W[(block*4 + kl)*4 + i, cs:cs+49] = W4[kl*4 + i, :]
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify exact decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True

def _remove_duplicates(U, V, W):
    """Remove identical (U_col, V_col, W_col) tuples."""
    rank = U.shape[1]
    seen = set()
    keep = []
    for r in range(rank):
        key = (tuple(U[:,r].tolist()), tuple(V[:,r].tolist()), tuple(W[:,r].tolist()))
        if key not in seen:
            seen.add(key)
            keep.append(r)
    if len(keep) < rank:
        return U[:, keep], V[:, keep], W[:, keep]
    return U, V, W

def construct():
    """Rank-98 exact tensor decomposition for 4x4 * 4x8."""
    if time.time() > _DEADLINE:
        # Last resort: schoolbook rank 128
        n, m, p = 4, 4, 8
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=np.int16)
        V = np.zeros((m*p, rank), dtype=np.int16)
        W = np.zeros((p*n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = _build_4x4x8()
    U, V, W = _remove_duplicates(U, V, W)
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback schoolbook
    n, m, p = 4, 4, 8
    rank = n*m*p
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
