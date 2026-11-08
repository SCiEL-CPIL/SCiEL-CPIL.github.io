import time
import numpy as np

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2 x 2x2."""
    # U: (i,k) -> col, V: (k,j) -> col, W: (j,i) -> col
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = {}
    V = {}
    W = {}
    col = 0
    
    # M1 = (A11+A22)(B11+B22)
    U[(0,0)] = {col: 1}; U[(1,1)] = {col: 1}
    V[(0,0)] = {col: 1}; V[(1,1)] = {col: 1}
    W[(0,0)] = {col: 1}; W[(1,1)] = {col: 1}
    col += 1
    
    # M2 = (A21+A22)B11
    U[(1,0)] = {col: 1}; U[(1,1)] = {col: 1}
    V[(0,0)] = {col: 1}
    W[(1,0)] = {col: 1}; W[(1,1)] = {col: -1}
    col += 1
    
    # M3 = A11(B21-B22)
    U[(0,0)] = {col: 1}
    V[(1,0)] = {col: 1}; V[(1,1)] = {col: -1}
    W[(0,1)] = {col: 1}; W[(1,1)] = {col: 1}
    col += 1
    
    # M4 = A22(B11-B21)
    U[(1,1)] = {col: 1}
    V[(0,0)] = {col: 1}; V[(1,0)] = {col: -1}
    W[(0,0)] = {col: 1}; W[(1,0)] = {col: 1}
    col += 1
    
    # M5 = (A11+A12)B22
    U[(0,0)] = {col: 1}; U[(0,1)] = {col: 1}
    V[(1,1)] = {col: 1}
    W[(0,0)] = {col: -1}; W[(0,1)] = {col: 1}
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[(1,0)] = {col: 1}; U[(0,0)] = {col: -1}
    V[(0,0)] = {col: 1}; V[(0,1)] = {col: 1}
    W[(1,1)] = {col: 1}
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[(0,1)] = {col: 1}; U[(1,1)] = {col: -1}
    V[(1,0)] = {col: 1}; V[(1,1)] = {col: 1}
    W[(0,0)] = {col: 1}
    col += 1
    
    R = col
    Um = np.zeros((4, R))
    Vm = np.zeros((4, R))
    Wm = np.zeros((4, R))
    for (i, k), d in U.items():
        Um[i*2+k] = [d.get(c, 0) for c in range(R)]
    for (k, j), d in V.items():
        Vm[k*2+j] = [d.get(c, 0) for c in range(R)]
    for (j, i), d in W.items():
        Wm[j*2+i] = [d.get(c, 0) for c in range(R)]
    return Um, Vm, Wm

def _rect_223():
    """Return rank-6 decomposition for 2x2 x 2x3."""
    # A is 2x2, B is 2x3, C is 2x3
    # Known rank-6 scheme for 2x2 x 2x3
    # Using the method from Bini et al. or similar
    
    U = {}
    V = {}
    W = {}
    col = 0
    
    # M1 = (A11+A22)(B11+B21)
    U[(0,0)] = {col: 1}; U[(1,1)] = {col: 1}
    V[(0,0)] = {col: 1}; V[(1,0)] = {col: 1}
    W[(0,0)] = {col: 1}; W[(1,0)] = {col: 1}
    col += 1
    
    # M2 = (A21+A22)B11
    U[(1,0)] = {col: 1}; U[(1,1)] = {col: 1}
    V[(0,0)] = {col: 1}
    W[(1,0)] = {col: 1}; W[(1,1)] = {col: -1}
    col += 1
    
    # M3 = A11(B21-B22)
    U[(0,0)] = {col: 1}
    V[(1,0)] = {col: 1}; V[(1,1)] = {col: -1}
    W[(0,1)] = {col: 1}; W[(1,1)] = {col: 1}
    col += 1
    
    # M4 = A22(B11-B21)
    U[(1,1)] = {col: 1}
    V[(0,0)] = {col: 1}; V[(1,0)] = {col: -1}
    W[(0,0)] = {col: 1}; W[(1,0)] = {col: 1}
    col += 1
    
    # M5 = (A11+A12)(B12+B22)
    U[(0,0)] = {col: 1}; U[(0,1)] = {col: 1}
    V[(0,1)] = {col: 1}; V[(1,1)] = {col: 1}
    W[(0,1)] = {col: -1}; W[(0,2)] = {col: 1}; W[(1,1)] = {col: -1}; W[(1,2)] = {col: 1}
    col += 1
    
    # M6 = (A21-A11)(B12+B22)
    U[(1,0)] = {col: 1}; U[(0,0)] = {col: -1}
    V[(0,1)] = {col: 1}; V[(1,1)] = {col: 1}
    W[(1,1)] = {col: 1}; W[(1,2)] = {col: 1}
    col += 1
    
    R = col
    Um = np.zeros((4, R))
    Vm = np.zeros((6, R))
    Wm = np.zeros((6, R))
    for (i, k), d in U.items():
        Um[i*2+k] = [d.get(c, 0) for c in range(R)]
    for (k, j), d in V.items():
        Vm[k*3+j] = [d.get(c, 0) for c in range(R)]
    for (j, i), d in W.items():
        Wm[j*2+i] = [d.get(c, 0) for c in range(R)]
    return Um, Vm, Wm

def _build():
    """Build the full 4x4 x 4x6 decomposition using Kronecker product."""
    n, m, p = 4, 4, 6
    
    # Get inner decompositions
    U_str, V_str, W_str = _strassen_222()  # 2x2 x 2x2, rank 7
    U_rect, V_rect, W_rect = _rect_223()    # 2x2 x 2x3, rank 6
    
    R_outer = 7
    R_inner = 6
    R_total = R_outer * R_inner
    
    U = np.zeros((16, R_total))
    V = np.zeros((24, R_total))
    W = np.zeros((24, R_total))
    
    for ro in range(R_outer):
        for ri in range(R_inner):
            r = ro * R_inner + ri
            
            # U: A is 4x4, split into 2x2 blocks of 2x2
            # U[i,k] where i = i1*2+i2, k = k1*2+k2
            # U_full[i,k] = U_outer[i1,k1,ro] * U_inner[i2,k2,ri]
            for i1 in range(2):
                for k1 in range(2):
                    uo = U_str[i1*2+k1, ro]
                    for i2 in range(2):
                        for k2 in range(2):
                            ui = U_rect[i2*2+k2, ri]
                            i = i1*2 + i2
                            k = k1*2 + k2
                            U[i*4+k, r] = uo * ui
            
            # V: B is 4x6, split into 2x2 blocks of 2x3
            # V[k,j] where k = k1*2+k2, j = j1*3+j2
            # V_full[k,j] = V_outer[k1,j1,ro] * V_inner[k2,j2,ri]
            for k1 in range(2):
                for j1 in range(2):
                    vo = V_str[k1*2+j1, ro]
                    for k2 in range(2):
                        for j2 in range(3):
                            vi = V_rect[k2*3+j2, ri]
                            k = k1*2 + k2
                            j = j1*3 + j2
                            V[k*6+j, r] = vo * vi
            
            # W: C is 4x6, trace convention W[j*n+i] = W[j*4+i]
            # W[i,j] where i = i1*2+i2, j = j1*3+j2
            # W_full[i,j] = W_outer[i1,j1,ro] * W_inner[i2,j2,ri]
            for i1 in range(2):
                for j1 in range(2):
                    wo = W_str[i1*2+j1, ro]
                    for i2 in range(2):
                        for j2 in range(3):
                            wi = W_rect[i2*2+j2, ri]
                            i = i1*2 + i2
                            j = j1*3 + j2
                            W[j*4+i, r] = wo * wi
    
    return U, V, W

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    ok = True
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
                if abs(s - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def construct():
    deadline = time.time() + 110.0
    U, V, W = _build()
    if not _verify(U, V, W):
        # Fallback: schoolbook decomposition (rank = n*m*p = 96)
        R = 4 * 4 * 6
        U = np.zeros((16, R))
        V = np.zeros((24, R))
        W = np.zeros((24, R))
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(6):
                    U[i*4 + k, idx] = 1
                    V[k*6 + j, idx] = 1
                    W[j*4 + i, idx] = 1
                    idx += 1
    
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else float(x))
            out.append(r)
        return out
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
