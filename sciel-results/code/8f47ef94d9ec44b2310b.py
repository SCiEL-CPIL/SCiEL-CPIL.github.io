"""Matrix multiplication 2x4x4 rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_28():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen(x_off, y_off, z_off, base_col):
        x0 = x_off * 2
        x1 = x_off * 2 + 1
        y0 = y_off * 2
        y1 = y_off * 2 + 1
        yc0 = z_off * 2
        yc1 = z_off * 2 + 1
        
        # Each of the 7 Strassen products gets its own column
        # M1 = (X11+X22)*(Y11+Y22), contributes Z11:+1, Z22:+1
        c = base_col
        U[0*m+x0][c] += 1; U[1*m+x1][c] += 1
        V[y0*p+yc0][c] += 1; V[y1*p+yc1][c] += 1
        W[yc0*n+0][c] += 1; W[yc1*n+1][c] += 1
        
        # M2 = (X21+X22)*Y11, contributes Z21:+1, Z22:-1
        c = base_col + 1
        U[1*m+x0][c] += 1; U[1*m+x1][c] += 1
        V[y0*p+yc0][c] += 1
        W[yc0*n+1][c] += 1; W[yc1*n+1][c] += -1
        
        # M3 = X11*(Y12-Y22), contributes Z12:+1, Z22:+1
        c = base_col + 2
        U[0*m+x0][c] += 1
        V[y0*p+yc1][c] += 1; V[y1*p+yc1][c] += -1
        W[yc1*n+0][c] += 1; W[yc1*n+1][c] += 1
        
        # M4 = X22*(Y21-Y11), contributes Z11:+1, Z21:+1
        c = base_col + 3
        U[1*m+x1][c] += 1
        V[y1*p+yc0][c] += 1; V[y0*p+yc0][c] += -1
        W[yc0*n+0][c] += 1; W[yc0*n+1][c] += 1
        
        # M5 = (X11+X12)*Y22, contributes Z11:-1, Z12:+1
        c = base_col + 4
        U[0*m+x0][c] += 1; U[0*m+x1][c] += 1
        V[y1*p+yc1][c] += 1
        W[yc0*n+0][c] += -1; W[yc1*n+0][c] += 1
        
        # M6 = (X21-X11)*(Y11+Y12), contributes Z22:+1
        c = base_col + 5
        U[1*m+x0][c] += 1; U[0*m+x0][c] += -1
        V[y0*p+yc0][c] += 1; V[y0*p+yc1][c] += 1
        W[yc1*n+1][c] += 1
        
        # M7 = (X11-X22)*(Y21+Y22), contributes Z11:+1, Z21:-1
        c = base_col + 6
        U[0*m+x0][c] += 1; U[1*m+x1][c] += -1
        V[y1*p+yc0][c] += 1; V[y1*p+yc1][c] += 1
        W[yc0*n+0][c] += 1; W[yc0*n+1][c] += -1
    
    # C[:,0:2] = A1*B11 + A2*B21
    add_strassen(0, 0, 0, 0)   # A1*B11
    add_strassen(1, 1, 0, 7)   # A2*B21
    
    # C[:,2:4] = A1*B12 + A2*B22
    add_strassen(0, 0, 1, 14)  # A1*B12
    add_strassen(1, 1, 1, 21)  # A2*B22
    
    return {"U": U, "V": V, "W": W}

def _try_search(rank, time_budget):
    n, m, p = 2, 4, 4
    start_time = time.time()
    
    for attempt in range(100):
        if time.time() - start_time > time_budget:
            break
        
        U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        for step in range(3000):
            if time.time() - start_time > time_budget:
                break
            mat = random.choice([U, V, W])
            r = random.randrange(len(mat))
            c = random.randrange(rank)
            old = mat[r][c]
            new = random.choice([-1, -0.5, 0, 0.5, 1])
            if new == old:
                continue
            mat[r][c] = new
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            mat[r][c] = old
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best = _strassen_28()
    
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 10:
        result = _try_search(27, remaining)
        if result is not None:
            return result
    
    return best
