import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE_FRAC = 0.90
START = time.time()

def _deadline():
    return time.time() - START > 150.0 * DEADLINE_FRAC

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)

def _make_2x2x2_strassen():
    """
    Returns U(4,7), V(4,7), W(4,7) for 2x2x2 Strassen.
    A indices: (i,j) -> flat i*2+j
    B indices: (j,k) -> flat j*2+k  
    C indices: (i,k) -> flat i*2+k, but W uses (k*n+i) convention so W[k*2+i]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)  
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 + M5 + M7
    # C01 = M5 + M6
    # C10 = M3 + M5
    # C11 = M2 + M3 + M4
    
    # M1: A has A00,A11; B has B00,B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: A has A10,A11; B has B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A has A00; B has B10-B11
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4: A has A11; B has B01-B00
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5: A has A00,A01; B has B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A has A10-A00; B has B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7: A has A01-A11; B has B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _make_4x4x4_strassen():
    """
    Build 4x4x4 Strassen (rank 49) by 2-level block decomposition.
    A(4x4) -> 2x2 blocks of 2x2
    B(4x4) -> 2x2 blocks of 2x2
    C(4x4) -> 2x2 blocks of 2x2
    
    Top-level: 7 Strassen products, each is a 2x2x2 multiplication (rank 7)
    Total: 7*7 = 49
    """
    n = m = p = 4
    R = 49
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    inner_U, inner_V, inner_W = _make_2x2x2_strassen()
    
    # Top-level Strassen coefficients
    # For each of 7 top products t, define which A-blocks and B-blocks participate
    # A-blocks indexed by (br, bc) where br, bc in {0,1}
    # B-blocks indexed by (bc2, bk) where bc2, bk in {0,1}
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    top_A = [
        [(0, 0, 1), (1, 1, 1)],   # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],   # M2: A10 + A11
        [(0, 0, 1)],              # M3: A00
        [(1, 1, 1)],              # M4: A11
        [(0, 0, 1), (0, 1, 1)],   # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],  # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)],  # M7: A01 - A11
    ]
    top_B = [
        [(0, 0, 1), (1, 1, 1)],   # M1: B00 + B11
        [(0, 0, 1)],              # M2: B00
        [(1, 0, 1), (1, 1, -1)],  # M3: B10 - B11
        [(0, 1, 1), (0, 0, -1)],  # M4: B01 - B00
        [(1, 1, 1)],              # M5: B11
        [(0, 0, 1), (0, 1, 1)],   # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)],   # M7: B10 + B11
    ]
    
    # C combination: which top products contribute to each C block
    # C00 = M1 + M4 + M5 + M7
    # C01 = M5 + M6
    # C10 = M3 + M5
    # C11 = M2 + M3 + M4
    top_C = [
        [0, 3, 4, 6],  # C00
        [4, 5],         # C01
        [2, 4],         # C10
        [1, 2, 3],      # C11
    ]
    
    for t in range(7):
        for tp in range(7):
            col = t * 7 + tp
            
            # Fill U: for each A-block (br,bc) with coefficient acoef
            for (br, bc, acoef) in top_A[t]:
                for ii in range(2):
                    for jj in range(2):
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        U[gi * m + gj, col] += acoef * inner_U[ii * 2 + jj, tp]
            
            # Fill V: for each B-block (bc2,bk) with coefficient bcoef
            for (bc2, bk, bcoef) in top_B[t]:
                for jj in range(2):
                    for kk in range(2):
                        gj = bc2 * 2 + jj
                        gk = bk * 2 + kk
                        V[gj * p + gk, col] += bcoef * inner_V[jj * 2 + kk, tp]
            
            # Fill W: for each C-block (cr,cc) that includes t
            for c_idx in range(4):
                if t in top_C[c_idx]:
                    cr, cc = c_idx // 2, c_idx % 2
                    for ii in range(2):
                        for kk in range(2):
                            gi = cr * 2 + ii
                            gk = cc * 2 + kk
                            W[gk * n + gi, col] += inner_W[kk * 2 + ii, tp]
    
    return U, V, W

def _strassen_444_5(n, m, p):
    """
    Build 4x4x5 decomposition:
    - Use 4x4x4 Strassen (rank 49) for k=0..3
    - Add 16 schoolbook columns for k=4
    Total rank: 49 + 16 = 65
    """
    U4, V4, W4 = _make_4x4x4_strassen()
    R4 = U4.shape[1]  # 49
    extra = n * m     # 16
    R = R4 + extra    # 65
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    U[:, :R4] = U4
    V[:, :R4] = V4
    W[:, :R4] = W4
    
    k = 4
    c = R4
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    best = None
    
    # Candidate 1: schoolbook (rank 80)
    Us, Vs, Ws = _schoolbook(n, m, p)
    if _verify(Us, Vs, Ws, n, m, p):
        best = (Us, Vs, Ws, Us.shape[1])
    
    # Candidate 2: Strassen 4x4x4 + schoolbook for k=4 (rank 65)
    if not _deadline():
        try:
            U2, V2, W2 = _strassen_444_5(n, m, p)
            if _verify(U2, V2, W2, n, m, p):
                if best is None or U2.shape[1] < best[3]:
                    best = (U2, V2, W2, U2.shape[1])
        except Exception:
            pass
    
    if best is None:
        best = (Us, Vs, Ws, Us.shape[1])
    
    U, V, W, _ = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
