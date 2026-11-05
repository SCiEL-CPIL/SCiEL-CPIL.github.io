import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140

    def strassen_2x2(i_off, j_off, k_off):
        """7 Strassen terms for 2x2 * 2x2 block at offsets."""
        terms = []
        u = [0]*(n*m); u[i_off*m+j_off] = 1; u[(i_off+1)*m+(j_off+1)] = 1
        v = [0]*(m*p); v[j_off*p+k_off] = 1; v[(j_off+1)*p+(k_off+1)] = 1
        w = [0]*(p*n); w[k_off*n+i_off] = 1; w[(k_off+1)*n+(i_off+1)] = 1
        terms.append((u, v, w))
        u = [0]*(n*m); u[(i_off+1)*m+j_off] = 1; u[(i_off+1)*m+(j_off+1)] = 1
        v = [0]*(m*p); v[j_off*p+k_off] = 1
        w = [0]*(p*n); w[k_off*n+(i_off+1)] = 1; w[(k_off+1)*n+(i_off+1)] = -1
        terms.append((u, v, w))
        u = [0]*(n*m); u[i_off*m+j_off] = 1
        v = [0]*(m*p); v[j_off*p+(k_off+1)] = 1; v[(j_off+1)*p+(k_off+1)] = -1
        w = [0]*(p*n); w[(k_off+1)*n+i_off] = 1; w[(k_off+1)*n+(i_off+1)] = 1
        terms.append((u, v, w))
        u = [0]*(n*m); u[(i_off+1)*m+(j_off+1)] = 1
        v = [0]*(m*p); v[(j_off+1)*p+k_off] = 1; v[j_off*p+k_off] = -1
        w = [0]*(p*n); w[k_off*n+i_off] = 1; w[k_off*n+(i_off+1)] = 1
        terms.append((u, v, w))
        u = [0]*(n*m); u[i_off*m+j_off] = 1; u[i_off*m+(j_off+1)] = 1
        v = [0]*(m*p); v[(j_off+1)*p+(k_off+1)] = 1
        w = [0]*(p*n); w[(k_off+1)*n+i_off] = 1; w[k_off*n+i_off] = -1
        terms.append((u, v, w))
        u = [0]*(n*m); u[(i_off+1)*m+j_off] = 1; u[i_off*m+j_off] = -1
        v = [0]*(m*p); v[j_off*p+k_off] = 1; v[j_off*p+(k_off+1)] = 1
        w = [0]*(p*n); w[(k_off+1)*n+(i_off+1)] = 1
        terms.append((u, v, w))
        u = [0]*(n*m); u[i_off*m+(j_off+1)] = 1; u[(i_off+1)*m+(j_off+1)] = -1
        v = [0]*(m*p); v[(j_off+1)*p+k_off] = 1; v[(j_off+1)*p+(k_off+1)] = 1
        w = [0]*(p*n); w[k_off*n+i_off] = 1
        terms.append((u, v, w))
        return terms

    all_terms = []
    
    # 4x4 * 4x4 part (columns 0-3 of B): 8 Strassen 2x2 blocks = 56 terms
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                all_terms.extend(strassen_2x2(i_off, j_off, k_off))
    
    # 4x4 * 4x3 part (columns 4-6 of B), schoolbook = 12 terms
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0]*(n*m)
                u[i*m+j] = 1
                v = [0]*(m*p)
                v[j*p+k] = 1
                w = [0]*(p*n)
                w[k*n+i] = 1
                all_terms.append((u, v, w))
    
    rank = len(all_terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for r in range(n*m):
            U[r][c] = u[r]
        for r in range(m*p):
            V[r][c] = v[r]
        for r in range(p*n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
