import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    Strategy:
    1. Split the multiplication based on the middle dimension m=3.
       Let j=0,1 be handled by M_{2,2,5} and j=2 by M_{2,1,5}.
    2. M_{2,1,5} has rank 10 (schoolbook).
    3. M_{2,2,5} is split into M_{2,2,2} (k=0,1), M_{2,2,2} (k=2,3), M_{2,2,1} (k=4).
       Rank = 7 (Strassen) + 7 (Strassen) + 4 (Schoolbook) = 18.
    4. Total Rank = 18 + 10 = 28.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m2
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        # m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        # m5
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        # m6
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
