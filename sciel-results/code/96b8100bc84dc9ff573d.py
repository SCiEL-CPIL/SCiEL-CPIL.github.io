"""Rewritten program for 3x5x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    def build_baseline():
        """Build the rank-99 baseline using 6 Strassen blocks + schoolbook."""
        rank = 99
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0*m+j0, col] += 1
            U[(i0+1)*m+j0+1, col] += 1
            V[j0*p+k0, col] += 1
            V[(j0+1)*p+k0+1, col] += 1
            W[k0*n+i0, col] += 1
            W[(k0+1)*n+i0+1, col] += 1
            col += 1
            U[(i0+1)*m+j0, col] += 1
            U[(i0+1)*m+j0+1, col] += 1
            V[j0*p+k0, col] += 1
            W[k0*n+i0+1, col] += 1
            W[(k0+1)*n+i0+1, col] -= 1
            col += 1
            U[i0*m+j0, col] += 1
            V[j0*p+k0+1, col] += 1
            V[(j0+1)*p+k0+1, col] -= 1
            W[(k0+1)*n+i0, col] += 1
            W[(k0+1)*n+i0+1, col] += 1
            col += 1
            U[(i0+1)*m+j0+1, col] += 1
            V[(j0+1)*p+k0, col] += 1
            V[j0*p+k0, col] -= 1
            W[k0*n+i0, col] += 1
            W[k0*n+i0+1, col] += 1
            col += 1
            U[i0*m+j0, col] += 1
            U[i0*m+j0+1, col] += 1
            V[(j0+1)*p+k0+1, col] += 1
            W[(k0+1)*n+i0, col] += 1
            W[k0*n+i0, col] -= 1
            col += 1
            U[(i0+1)*m+j0, col] += 1
            U[i0*m+j0, col] -= 1
            V[j0*p+k0, col] += 1
            V[j0*p+k0+1, col] += 1
            W[(k0+1)*n+i0+1, col] += 1
            col += 1
            U[(i0+1)*m+j0+1, col] += 1
            U[i0*m+j0+1, col] -= 1
            V[(j0+1)*p+k0, col] += 1
            V[(j0+1)*p+k0+1, col] += 1
            W[k0*n+i0, col] += 1
            col += 1
        
        def add_schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)
        add_schoolbook([2], range(5), range(7))
        add_schoolbook([0, 1], [4], range(7))
        add_schoolbook([0, 1], [0, 1, 2, 3], [6])
        
        return U, V, W, col
    
    U, V, W, actual_rank = build_baseline()
    
    # Quick verification
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entries.append((a, b, c))
    
    valid = True
    for a, b, c in entries:
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j == j2 and k == k2 and i == i2:
            computed = float(np.dot(np.dot(U[a, :actual_rank], V[b, :actual_rank]), W[c, :actual_rank]))
            if abs(computed - 1.0) > 1e-8:
                valid = False
                break
    
    if not valid:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
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
