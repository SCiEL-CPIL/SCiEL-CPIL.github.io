import time
import numpy as np

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
DEADLINE = START_TIME + 140.0

def construct():
    n, m, p = 2, 3, 5
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    
    # Helper to add Strassen block
    def add_strassen(j_off, k_off):
        nonlocal c
        # 7 terms for 2x2x2 Strassen
        # Local indices i in {0,1}, j in {0,1}, k in {0,1}
        # Global indices:
        # I = i*m + (j_off + j)
        # J = (j_off + j)*p + (k_off + k)
        # K = (k_off + k)*n + i
        
        # 1. (A00+A11)(B00+B11)
        U[0*m + (j_off+0)][c] = 1
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # 2. (A10+A11)(B00)
        U[1*m + (j_off+0)][c] = 1
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        W[(k_off+0)*n + 1][c] = 1
        W[(k_off+1)*n + 1][c] = -1
        c += 1
        
        # 3. (A00)(B01 - B11)
        U[0*m + (j_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # 4. (A11)(B10 - B00)
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+0)][c] = -1
        W[(k_off+0)*n + 0][c] = 1
        W[(k_off+0)*n + 1][c] = 1
        c += 1
        
        # 5. (A00+A01)(B11)
        U[0*m + (j_off+0)][c] = 1
        U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + 0][c] = -1
        W[(k_off+1)*n + 0][c] = 1
        c += 1
        
        # 6. (A10-A00)(B00+B01)
        U[1*m + (j_off+0)][c] = 1
        U[0*m + (j_off+0)][c] = -1
        V[(j_off+0)*p + (k_off+0)][c] = 1
        V[(j_off+0)*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        
        # 7. (A01-A11)(B10+B11)
        U[0*m + (j_off+1)][c] = 1
        U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + (k_off+0)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[(k_off+0)*n + 0][c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen(0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen(0, 2)
    
    # Block 3: j in {0,1}, k in {4}
    # Rank 1 term
    for i in range(2):
        for j in range(2):
            U[i*m + j][c] = 1
    for j in range(2):
        V[j*p + 4][c] = 1
    for i in range(2):
        W[4*n + i][c] = 1
    c += 1
    
    # Block 4: j in {2}, k in {0..4}
    # Rank 1 term
    for i in range(2):
        U[i*m + 2][c] = 1
    for k in range(5):
        V[2*p + k][c] = 1
    for k in range(5):
        for i in range(2):
            W[k*n + i][c] = 1
    c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True
