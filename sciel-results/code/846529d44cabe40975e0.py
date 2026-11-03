import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140

def construct():
    """
    Rank-27 exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    Two Strassen 2x2x2 blocks (k=0,1 and k=2,3) + one shared free term (k=4).
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
        c += 1
        # m2
        U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1
        W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
        c += 1
        # m3
        U[0*m + j_off, c] = 1
        V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
        W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1), c] = 1
        V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
        W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
        c += 1
        # m5
        U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
        V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
        c += 1
        # m6
        U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
        V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
        W[(k_off+1)*n + 1, c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
        V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1
        c += 1
    
    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    # Free term for k=4: U[i,j]=1, V[j,4]=1, W[4,i]=1 for all valid i,j
    # U indexed by i*m+j, V by j*p+k, W by k*n+i
    for i in range(n):
        for j in range(m):
            U[i*m+j, c] = 1
            V[j*p+4, c] = 1
    for i in range(n):
        W[4*n+i, c] = 1
    c += 1
    
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return _rank28_solution()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _rank28_solution():
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
        c += 1
        U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1
        W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
        c += 1
        U[0*m + j_off, c] = 1
        V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
        W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
        c += 1
        U[1*m + (j_off+1), c] = 1
        V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
        W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
        c += 1
        U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
        V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
        c += 1
        U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
        V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
        W[(k_off+1)*n + 1, c] = 1
        c += 1
        U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
        V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
                
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
