import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 4
    rank = 34
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strassen Left (A11 * B1L) - indices 0-4 in U, 0-5 in V, 0-4 in W
    # P1: (A11+A22)*(B11+B22)
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[0, col] = 1; W[4, col] = 1
    col += 1
    
    # P2: (A21+A22)*B11
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[4, col] = -1
    col += 1
    
    # P3: A11*(B21-B11)
    U[0, col] = 1
    V[1, col] = 1; V[5, col] = -1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # P4: A22*(B12-B22)
    U[4, col] = 1
    V[4, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # P5: (A11+A12)*B22
    U[0, col] = 1; U[1, col] = 1
    V[5, col] = 1
    W[0, col] = -1; W[3, col] = 1
    col += 1
    
    # P6: (A21-A11)*(B11+B12)
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[4, col] = 1
    col += 1
    
    # P7: (A12-A22)*(B21+B22)
    U[1, col] = 1; U[4, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[0, col] = 1
    col += 1
    
    # A12 * B2L - direct multiplications
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    
    # A21 * B1L - direct multiplications
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    
    # A22 * B2L - direct multiplications
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    
    # Strassen Right (A11 * B1R) - indices 2-7 in V, 6-11 in W
    # P1: (A11+A22)*(B11+B22)
    U[0, col] = 1; U[4, col] = 1
    V[2, col] = 1; V[7, col] = 1
    W[6, col] = 1; W[10, col] = 1
    col += 1
    
    # P2: (A21+A22)*B11
    U[3, col] = 1; U[4, col] = 1
    V[2, col] = 1
    W[7, col] = 1; W[10, col] = -1
    col += 1
    
    # P3: A11*(B21-B11)
    U[0, col] = 1
    V[3, col] = 1; V[7, col] = -1
    W[9, col] = 1; W[10, col] = 1
    col += 1
    
    # P4: A22*(B12-B22)
    U[4, col] = 1
    V[6, col] = 1; V[2, col] = -1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    # P5: (A11+A12)*B22
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1
    W[6, col] = -1; W[9, col] = 1
    col += 1
    
    # P6: (A21-A11)*(B11+B12)
    U[3, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[10, col] = 1
    col += 1
    
    # P7: (A12-A22)*(B21+B22)
    U[1, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[6, col] = 1
    col += 1
    
    # A12 * B2R - direct multiplications
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    
    # A21 * B1R - direct multiplications
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    
    # A22 * B2R - direct multiplications
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1
    
    # Verify correctness
    T_reconstructed = np.zeros((9, 12, 12), dtype=int)
    for r in range(rank):
        T_reconstructed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(4):
                T_expected[i*3+j, j*4+k, k*3+i] = 1
    
    if not np.array_equal(T_reconstructed, T_expected):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
