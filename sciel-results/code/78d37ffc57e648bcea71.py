import time
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M2 = (c+d)e
        U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        # M3 = a(f-h)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M4 = d(g-e)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
        col += 1
        # M5 = (a+b)h
        U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
        col += 1
        # M6 = (c-a)(e+f)
        U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        # M7 = (b-d)(g+h)
        U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    def add_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj), col] = 1
                V[(j0+dj)*p + k0, col] = 1
                W[k0*n + (i0+di), col] = 1
                col += 1

    # A cols 0,1 (j0=0):
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_schoolbook_2x2x1(0, 0, 6)
    
    # A cols 2,3 (j0=2):
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    add_schoolbook_2x2x1(0, 2, 6)

    if time.time() - start_time > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
