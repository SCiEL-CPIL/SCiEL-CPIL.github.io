"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M0 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M1 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # M2 = A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # M3 = A11(B10-B00)
    U[3,3]=1
    V[0,3]=-1; V[2,3]=1
    W[0,3]=1; W[1,3]=1
    
    # M4 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # M5 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M6 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _schoolbook_3x3():
    """Rank-27 schoolbook for 3x3x3."""
    U = np.zeros((9, 27), dtype=int)
    V = np.zeros((9, 27), dtype=int)
    W = np.zeros((9, 27), dtype=int)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    return U, V, W

def _kronecker_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions.
    If (U1,V1,W1) decomposes n1×m1×p1 and (U2,V2,W2) decomposes n2×m2×p2,
    then the Kronecker decomposes (n1*n2)×(m1*m2)×(p1*p2).
    """
    n1, m1, p1 = U1.shape[0], V1.shape[0], W1.shape[0]
    # Actually U1 has shape (n1*m1, r1), V1 (m1*p1, r1), W1 (p1*n1, r1)
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    n2, m2, p2 = U2.shape[0], V2.shape[0], W2.shape[0]
    # U2 shape (n2*m2, r2), V2 (m2*p2, r2), W2 (p2*n2, r2)
    
    rank = r1 * r2
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: row index i*m + j where i = i1*n2 + i2, j = j1*m2 + j2
            # U1[i1*m1 + j1, c1] * U2[i2*m2 + j2, c2]
            for i1 in range(n1):
                for j1 in range(m1):
                    v1 = U1[i1*m1 + j1, c1]
                    if v1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            v2 = U2[i2*m2 + j2, c2]
                            if v2 == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, c] += v1 * v2
            
            # V: row index j*p + k where j = j1*m2 + j2, k = k1*p2 + k2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V1[j1*p1 + k1, c1]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2*p2 + k2, c2]
                            if v2 == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, c] += v1 * v2
            
            # W: row index k*n + i where k = k1*p2 + k2, i = i1*n2 + i2
            for k1 in range(p1):
                for i1 in range(n1):
                    v1 = W1[k1*n1 + i1, c1]
                    if v1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            v2 = W2[k2*n2 + i2, c2]
                            if v2 == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, c] += v1 * v2
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, sample=True):
    """Verify the decomposition. If sample=True, check a subset."""
    rank = U.shape[1]
    if sample:
        rng = np.random.default_rng(42)
        indices = [(int(rng.integers(0,n)), int(rng.integers(0,m)), int(rng.integers(0,p))) for _ in range(100)]
    else:
        indices = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    
    for (i, j, k) in indices:
        s = 0.0
        for c in range(rank):
            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        if abs(s - 1.0) > 1e-9:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Approach: Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27)
    # This gives 6x6x6 with rank 7*27 = 189
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()
    
    U, V, W = _kronecker_decomp(U2, V2, W2, U3, V3, W3)
    
    # Verify
    if _verify_decomp(U, V, W, 6, 6, 6, sample=True):
        if _verify_decomp(U, V, W, 6, 6, 6, sample=False):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: direct block decomposition (3x3 grid of 2x2 Strassen)
    rank = 27 * 7
    Uf = np.zeros((36, rank), dtype=int)
    Vf = np.zeros((36, rank), dtype=int)
    Wf = np.zeros((36, rank), dtype=int)
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            Uf[u_row, col] = U2[li*2+lj, c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            Vf[v_row, col] = V2[lj*2+lk, c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            Wf[w_row, col] = W2[lk*2+li, c]
                    
                    col += 1
    
    if _verify_decomp(Uf, Vf, Wf, 6, 6, 6, sample=False):
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    
    # Schoolbook fallback
    rank_sb = 216
    Us = np.zeros((36, rank_sb), dtype=int)
    Vs = np.zeros((36, rank_sb), dtype=int)
    Ws = np.zeros((36, rank_sb), dtype=int)
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                Us[i*6+j, col] = 1
                Vs[j*6+k, col] = 1
                Ws[k*6+i, col] = 1
                col += 1
    
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
