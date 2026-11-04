import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
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
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
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
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition:
    - Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (7, Strassen)
    - Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (7, Strassen)
    - Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (4, Schoolbook)
    - Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (10, Schoolbook)
    Total: 28
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,5,6
    # C indices: 0,1,2,3
    a_map1 = [0, 1, 3, 4]
    b_map1 = [0, 1, 5, 6]
    c_map1 = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map1[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map1[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map1[s]][col] += W_s[s][r]
        col += 1
        
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # A indices: 0,1,3,4
    # B indices: 2,3,7,8
    # C indices: 4,5,6,7
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 7, 8]
    c_map2 = [4, 5, 6, 7]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map2[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map2[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map2[s]][col] += W_s[s][r]
        col += 1

    # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    # A indices: 0,1,3,4
    # B indices: 4,9
    # C indices: 8,9
    # Terms:
    # 1: A[0,0]*B[0,4] -> C[0,4] (W[8])
    # 2: A[0,1]*B[1,4] -> C[0,4] (W[8])
    # 3: A[1,0]*B[0,4] -> C[1,4] (W[9])
    # 4: A[1,1]*B[1,4] -> C[1,4] (W[9])
    
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    U[1][col] = 1; V[9][col] = 1; W[8][col] = 1; col += 1
    U[3][col] = 1; V[4][col] = 1; W[9][col] = 1; col += 1
    U[4][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1

    # Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    # A indices: 2,5
    # B indices: 10,11,12,13,14
    # C indices: 0..9
    # Terms: A[i,2]*B[2,k] -> C[i,k]
    # C index: k*2 + i
    
    for k in range(5):
        # i=0
        U[2][col] = 1; V[10+k][col] = 1; W[k*2+0][col] = 1; col += 1
        # i=1
        U[5][col] = 1; V[10+k][col] = 1; W[k*2+1][col] = 1; col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    result = _rank28_decomposition(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
    
    if valid:
        return result
    
    # Fallback to schoolbook if 28 fails
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
