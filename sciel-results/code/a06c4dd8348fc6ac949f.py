import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _check_validity(U, V, W, n, m, p, rank):
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k2 in range(p):
                            expected = 1.0 if (i2 == i and j == j2 and k2 == k) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += W[k*n+i, c] * U[i2*m+j, c] * V[j2*p+k2, c]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # Rank 28 construction
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i_off, j_off, k_off are offsets
    # i in {0,1}, j in {0,1}, k in {0,1} relative to offset
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # P1 = (A00+A11)(B00+B11)
        U[i_off*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[k_off*n + i_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # P2 = (A10+A11)B00
        U[(i_off+1)*m + j_off, col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[j_off*p + k_off, col] = 1
        W[k_off*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # P3 = A00(B01-B11)
        U[i_off*m + j_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1
        
        # P4 = A11 B10
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + k_off, col] = 1
        W[k_off*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[i_off*m + j_off, col] = 1
        U[i_off*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[(i_off+1)*m + j_off, col] = 1
        U[i_off*m + j_off, col] = -1
        V[j_off*p + k_off, col] = 1
        V[j_off*p + (k_off+1), col] = 1
        W[k_off*n + (i_off+1), col] = 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[i_off*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + k_off, col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + i_off, col] = 1
        col += 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    
    # Remaining terms:
    # j=2, all i, k
    # j in {0,1}, i in {0,1}, k=4
    
    # j=2
    for i in range(n):
        for k in range(p):
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # j in {0,1}, k=4
    for i in range(n):
        for j in range(2):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1

    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook if logic error
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Check validity
    if not _check_validity(U, V, W, n, m, p, rank):
        # Fallback
        rank_school = n * m * p
        U = np.zeros((n*m, rank_school))
        V = np.zeros((m*p, rank_school))
        W = np.zeros((p*n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
