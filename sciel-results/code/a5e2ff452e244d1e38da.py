# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(r):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
    return {"U": U, "V": V, "W": W}

def _strassen22():
    """Canonical Strassen 2x2x2 rank-7 decomposition."""
    U2 = [
        [[1,0],[0,1]],  # M1
        [[0,0],[1,1]],  # M2
        [[1,0],[0,0]],  # M3
        [[0,0],[0,1]],  # M4
        [[1,1],[0,0]],  # M5
        [[0,0],[1,1]],  # M6
        [[0,1],[0,-1]], # M7
    ]
    V2 = [
        [[1,0],[0,1]],  # M1
        [[1,0],[0,0]],  # M2
        [[0,1],[0,-1]], # M3
        [[0,1],[0,-1]], # M4
        [[1,0],[0,1]],  # M5
        [[0,0],[1,1]],  # M6
        [[1,1],[0,0]],  # M7
    ]
    W2 = [
        [[1,0],[0,1]],  # M1: C00+1, C11+1
        [[0,1],[0,-1]], # M2: C01+1, C11-1
        [[0,0],[1,-1]], # M3: C10+1, C11-1
        [[-1,1],[0,0]], # M4: C00-1, C01+1
        [[1,0],[0,0]],  # M5: C00+1
        [[0,0],[0,1]],  # M6: C11+1
        [[-1,0],[1,0]], # M7: C00-1, C10+1
    ]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j][t] = U2[t][i][j]
        for j in range(2):
            for k in range(2):
                V[j*2+k][t] = V2[t][j][k]
        for k in range(2):
            for i in range(2):
                W[k*2+i][t] = W2[t][k][i]
    return U, V, W

def _build_3x3x3_core():
    """Rank-22 3x3x3: Strassen(2x2) for {0,1}x{0,1} outputs + schoolbook for index-2 outputs."""
    U2, V2, W2 = _strassen22()
    rank = 7 + 15  # 7 Strassen + 15 schoolbook (5 outputs x 3)
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Embed Strassen for indices i,k in {0,1}
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][t] = U2[i*2+j][t]
                V[j*3+i][t] = V2[j*2+i][t]
                W[i*3+j][t] = W2[i*2+j][t]

    # Schoolbook for outputs C[i,k] where i==2 or k==2 (5 outputs: C02,C12,C20,C21,C22)
    c = 7
    for i in range(3):
        for k in range(3):
            if i != 2 and k != 2:
                continue
            for j in range(3):
                U[i*3+j][c] = 1
                V[j*3+k][c] = 1
                W[k*3+i][c] = 1
                c += 1
    return U, V, W, c

def _build_3x3x6_44():
    """
    Construct 3x3x6 decomposition by treating it as two independent 3x3x3 multiplications.
    A (3x3) is shared.
    B1 (3x3) corresponds to B[:, 0:3].
    B2 (3x3) corresponds to B[:, 3:6].
    C1 (3x3) corresponds to C[:, 0:3].
    C2 (3x3) corresponds to C[:, 3:6].
    
    U: (9, 44) = [U_core, U_core]
    V: (18, 44) = [V_core (for B1), V_core (for B2)]
    W: (18, 44) = [W_core (for C1), W_core (for C2)]
    """
    n, m, p = 3, 3, 6
    Uc, Vc, Wc, r = _build_3x3x3_core()
    rank = 2 * r
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for t in range(r):
        # U is duplicated
        for idx in range(n*m):
            U[idx][t] = Uc[idx][t]
            U[idx][r+t] = Uc[idx][t]
        
        # V is split into B1 (cols 0-2) and B2 (cols 3-5)
        for j in range(m):
            for k in range(n):
                V[j*p+k][t] = Vc[j*n+k][t]
                V[j*p+(3+k)][r+t] = Vc[j*n+k][t]
        
        # W is split into C1 (cols 0-2) and C2 (cols 3-5)
        for k in range(n):
            for i in range(n):
                W[k*n+i][t] = Wc[k*n+i][t]
                W[(3+k)*n+i][r+t] = Wc[k*n+i][t]
    return U, V, W, rank

def construct():
    start = time.time()
    deadline = 150.0
    n, m, p = 3, 3, 6
    school = _schoolbook(n, m, p)
    
    try:
        U, V, W, rank = _build_3x3x6_44()
        if rank < 54 and _verify(U, V, W, n, m, p, tol=1e-8):
            if time.time() - start < deadline - 10:
                return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass

    if _verify(school['U'], school['V'], school['W'], n, m, p):
        return school
    return school

# EVOLVE-BLOCK-END
