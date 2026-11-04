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
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = 1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _rank_24_decomposition():
    n, m, p = 2, 3, 5
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j in {0,1}, k in {0,1}
    U_s, V_s, W_s = _strassen_2x2x2()
    a_map = [0, 1, 3, 4] # A[0,0], A[0,1], A[1,0], A[1,1]
    b_map = [0, 1, 5, 6] # B[0,0], B[0,1], B[1,0], B[1,1]
    c_map = [0, 1, 2, 3] # C[0,0], C[0,1], C[1,0], C[1,1]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0: U[a_map[s]][col + r] += U_s[s][r]
            if V_s[s][r] != 0: V[b_map[s]][col + r] += V_s[s][r]
            if W_s[s][r] != 0: W[c_map[s]][col + r] += W_s[s][r]
    col += 7
    
    # Block 2: j in {0,1}, k in {2,3}
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 7, 8] # B[0,2], B[0,3], B[1,2], B[1,3]
    c_map2 = [4, 5, 6, 7] # C[2,0], C[2,1], C[3,0], C[3,1]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0: U[a_map2[s]][col + r] += U_s[s][r]
            if V_s[s][r] != 0: V[b_map2[s]][col + r] += V_s[s][r]
            if W_s[s][r] != 0: W[c_map2[s]][col + r] += W_s[s][r]
    col += 7
    
    # Block 3: j=2, k in {0..3}
    # Rank 4
    for r in range(4):
        k = r
        U[2][col + r] = 1  # A[0,2]
        U[5][col + r] = 1  # A[1,2]
        V[10 + k][col + r] = 1 # B[2,k]
        W[2*k][col + r] = 1     # C[k,0]
        W[2*k + 1][col + r] = 1 # C[k,1]
    col += 4
    
    # Block 4: j in {0,1,2}, k=4
    # Rank 6
    for i in range(2):
        for j in range(3):
            U[i * 3 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    
    try:
        result = _rank_24_decomposition()
        T = _build_target_tensor(n, m, p)
        valid, diff = _verify_decomposition(result["U"], result["V"], result["W"], T)
        if valid:
            return result
    except Exception:
        pass
        
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
