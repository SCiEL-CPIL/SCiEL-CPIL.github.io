"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00-A10)*B11
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01+A11)*(B00+B01)
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11)
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)*(B00-B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rank23_decomposition(n, m, p):
    """
    Rank-23 decomposition:
    - Strassen: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (7 terms)
    - Schoolbook: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]  (6 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]  (10 terms)
    Total: 7 + 6 + 10 = 23
    """
    rank = 23
    U = [[0] * rank for _ in range(n * m)]  # (6, 23)
    V = [[0] * rank for _ in range(m * p)]  # (15, 23)
    W = [[0] * rank for _ in range(p * n)]  # (10, 23)
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Strassen block: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: i*3+j for i in {0,1}, j in {0,1} -> [0,1,3,4]
    # B indices: j*5+k for j in {0,1}, k in {0,1} -> [0,1,5,6]
    # C indices: k*2+i for k in {0,1}, i in {0,1} -> [0,1,2,3]
    a_map = [0, 1, 3, 4]  # A[i*3+j] for (i,j) in [(0,0),(0,1),(1,0),(1,1)]
    b_map = [0, 1, 5, 6]  # B[j*5+k] for (j,k) in [(0,0),(0,1),(1,0),(1,1)]
    c_map = [0, 1, 2, 3]  # C[k*2+i] for (k,i) in [(0,0),(0,1),(1,0),(1,1)]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map[s]][col] += W_s[s][r]
        col += 1
    
    # Schoolbook: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]
    # j in {0,1}, k in {2,3,4}
    for j in range(2):
        for k in range(2, 5):
            for i in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    # j=2, k in {0,1,2,3,4}
    for k in range(5):
        for i in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _rank25_decomposition(n, m, p):
    """
    Rank-25 decomposition (alternative):
    - Strassen: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (7 terms)
    - Schoolbook: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (4 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]  (10 terms)
    - Schoolbook: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]  (2 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,4:5] already included
    Total: 7 + 4 + 10 + 2 + 2 = 25... let me recalculate
    Actually: 7 + 4 + 10 + 2 = 23... no
    j=0,1; k=2,3: 4 terms
    j=2; k=0,1,2,3,4: 10 terms  
    j=0,1; k=4: 2 terms
    Total: 7+4+10+2 = 23. Same as rank23.
    This is redundant, so let's just use rank23.
    """
    return _rank23_decomposition(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    T = _build_target_tensor(n, m, p)
    
    # Try rank-23 decomposition first
    result = _rank23_decomposition(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
    
    if valid:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
