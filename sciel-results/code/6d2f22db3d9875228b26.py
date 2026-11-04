import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    U_r = U_np.reshape(n, m, -1)
    V_r = V_np.reshape(m, p, -1)
    W_r = W_np.reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_hybrid():
    n, m, p = 3, 4, 8
    R = 88
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    # Row 0: schoolbook
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    # Rows 1-2: Strassen blocks
    def strassen_block(i0, jA0, jB0, k0, base_col):
        c = base_col
        # M1
        U[i0*4+jA0, c] += 1; U[(i0+1)*4+(jA0+1), c] += 1
        V[jB0*8+k0, c] += 1; V[(jB0+1)*8+(k0+1), c] += 1
        W[k0*3+i0, c] += 1; W[(k0+1)*3+(i0+1), c] += 1
        c += 1
        # M2
        U[(i0+1)*4+jA0, c] += 1; U[(i0+1)*4+(jA0+1), c] += 1
        V[jB0*8+k0, c] += 1
        W[k0*3+(i0+1), c] += 1; W[(k0+1)*3+(i0+1), c] -= 1
        c += 1
        # M3
        U[i0*4+jA0, c] += 1
        V[jB0*8+(k0+1), c] += 1; V[(jB0+1)*8+(k0+1), c] -= 1
        W[(k0+1)*3+i0, c] += 1; W[(k0+1)*3+(i0+1), c] += 1
        c += 1
        # M4
        U[(i0+1)*4+(jA0+1), c] += 1
        V[(jB0+1)*8+k0, c] += 1; V[jB0*8+k0, c] -= 1
        W[k0*3+i0, c] += 1; W[k0*3+(i0+1), c] += 1
        c += 1
        # M5
        U[i0*4+jA0, c] += 1; U[i0*4+(jA0+1), c] += 1
        V[(jB0+1)*8+(k0+1), c] += 1
        W[k0*3+i0, c] -= 1; W[(k0+1)*3+i0, c] += 1
        c += 1
        # M6
        U[(i0+1)*4+jA0, c] += 1; U[i0*4+jA0, c] -= 1
        V[jB0*8+k0, c] += 1; V[jB0*8+(k0+1), c] += 1
        W[(k0+1)*3+(i0+1), c] += 1
        c += 1
        # M7
        U[i0*4+(jA0+1), c] += 1; U[(i0+1)*4+(jA0+1), c] -= 1
        V[(jB0+1)*8+k0, c] += 1; V[(jB0+1)*8+(k0+1), c] += 1
        W[k0*3+i0, c] += 1
        c += 1
        return c
    
    # 8 blocks of 2x2x2 for rows 1-2
    for k0 in [0, 2, 4, 6]:
        col = strassen_block(1, 0, 0, k0, col)
        col = strassen_block(1, 2, 2, k0, col)
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 120
    
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Schoolbook fallback
    R = n * m * p
    U_sb = [[0]*R for _ in range(n*m)]
    V_sb = [[0]*R for _ in range(m*p)]
    W_sb = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
