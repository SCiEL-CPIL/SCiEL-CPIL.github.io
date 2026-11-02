import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n=None, m=None, p=None):
    if n is None:
        n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_2x2_strassen():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 0, 1, -1]
    W[:, 2] = [0, 1, 0, 1]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 1, 0, 0]
    W[:, 3] = [-1, 0, 1, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [1, 1, 0, 0]
    
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 1, 0]
    
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _build_2x2_times_2x4_strassen():
    U_s, V_s, W_s = _build_2x2_strassen()
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    for r in range(7):
        U[:, r] = U_s[:, r]
        V[0, r] = V_s[0, r]
        V[1, r] = V_s[1, r]
        V[4, r] = V_s[2, r]
        V[5, r] = V_s[3, r]
        W[0, r] = W_s[0, r]
        W[1, r] = W_s[1, r]
        W[2, r] = W_s[2, r]
        W[3, r] = W_s[3, r]
        
        U[:, 7 + r] = U_s[:, r]
        V[2, 7 + r] = V_s[0, r]
        V[3, 7 + r] = V_s[1, r]
        V[6, 7 + r] = V_s[2, r]
        V[7, 7 + r] = V_s[3, r]
        W[4, 7 + r] = W_s[0, r]
        W[5, 7 + r] = W_s[1, r]
        W[6, 7 + r] = W_s[2, r]
        W[7, 7 + r] = W_s[3, r]
        
    return U, V, W

def _build_2x4_times_4x4():
    U_22_24, V_22_24, W_22_24 = _build_2x2_times_2x4_strassen()
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    
    for r in range(14):
        U[0, r] = U_22_24[0, r]
        U[1, r] = U_22_24[1, r]
        U[4, r] = U_22_24[2, r]
        U[5, r] = U_22_24[3, r]
        V[0, r] = V_22_24[0, r]
        V[1, r] = V_22_24[1, r]
        V[2, r] = V_22_24[2, r]
        V[3, r] = V_22_24[3, r]
        V[4, r] = V_22_24[4, r]
        V[5, r] = V_22_24[5, r]
        V[6, r] = V_22_24[6, r]
        V[7, r] = V_22_24[7, r]
        W[0, r] = W_22_24[0, r]
        W[1, r] = W_22_24[1, r]
        W[2, r] = W_22_24[2, r]
        W[3, r] = W_22_24[3, r]
        W[4, r] = W_22_24[4, r]
        W[5, r] = W_22_24[5, r]
        W[6, r] = W_22_24[6, r]
        W[7, r] = W_22_24[7, r]
        
        U[2, 14 + r] = U_22_24[0, r]
        U[3, 14 + r] = U_22_24[1, r]
        U[6, 14 + r] = U_22_24[2, r]
        U[7, 14 + r] = U_22_24[3, r]
        V[8, 14 + r] = V_22_24[0, r]
        V[9, 14 + r] = V_22_24[1, r]
        V[10, 14 + r] = V_22_24[2, r]
        V[11, 14 + r] = V_22_24[3, r]
        V[12, 14 + r] = V_22_24[4, r]
        V[13, 14 + r] = V_22_24[5, r]
        V[14, 14 + r] = V_22_24[6, r]
        V[15, 14 + r] = V_22_24[7, r]
        W[0, 14 + r] = W_22_24[0, r]
        W[1, 14 + r] = W_22_24[1, r]
        W[2, 14 + r] = W_22_24[2, r]
        W[3, 14 + r] = W_22_24[3, r]
        W[4, 14 + r] = W_22_24[4, r]
        W[5, 14 + r] = W_22_24[5, r]
        W[6, 14 + r] = W_22_24[6, r]
        W[7, 14 + r] = W_22_24[7, r]
        
    return U, V, W

def _build_1x4_times_4x4_schoolbook():
    U = np.zeros((4, 16))
    V = np.zeros((16, 16))
    W = np.zeros((4, 16))
    col = 0
    for j in range(4):
        for k in range(4):
            U[j, col] = 1
            V[j * 4 + k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def _build_3x4_times_4x4():
    U_24, V_24, W_24 = _build_2x4_times_4x4()
    U_14, V_14, W_14 = _build_1x4_times_4x4_schoolbook()
    
    U = np.zeros((12, 44))
    V = np.zeros((16, 44))
    W = np.zeros((12, 44))
    
    for r in range(28):
        U[0, r] = U_24[0, r]
        U[1, r] = U_24[1, r]
        U[2, r] = U_24[2, r]
        U[3, r] = U_24[3, r]
        U[4, r] = U_24[4, r]
        U[5, r] = U_24[5, r]
        U[6, r] = U_24[6, r]
        U[7, r] = U_24[7, r]
        V[:, r] = V_24[:, r]
        W[0, r] = W_24[0, r]
        W[1, r] = W_24[1, r]
        W[3, r] = W_24[2, r]
        W[4, r] = W_24[3, r]
        W[6, r] = W_24[4, r]
        W[7, r] = W_24[5, r]
        W[9, r] = W_24[6, r]
        W[10, r] = W_24[7, r]
        
    for r in range(16):
        U[8, 28 + r] = U_14[0, r]
        U[9, 28 + r] = U_14[1, r]
        U[10, 28 + r] = U_14[2, r]
        U[11, 28 + r] = U_14[3, r]
        V[:, 28 + r] = V_14[:, r]
        W[2, 28 + r] = W_14[0, r]
        W[5, 28 + r] = W_14[1, r]
        W[8, 28 + r] = W_14[2, r]
        W[11, 28 + r] = W_14[3, r]
        
    return U, V, W

def _build_3x4_times_4x8():
    U_34_44, V_34_44, W_34_44 = _build_3x4_times_4x4()
    
    U = np.zeros((12, 88))
    V = np.zeros((32, 88))
    W = np.zeros((24, 88))
    
    for r in range(44):
        U[:, r] = U_34_44[:, r]
        V[0, r] = V_34_44[0, r]
        V[1, r] = V_34_44[1, r]
        V[2, r] = V_34_44[2, r]
        V[3, r] = V_34_44[3, r]
        V[8, r] = V_34_44[4, r]
        V[9, r] = V_34_44[5, r]
        V[10, r] = V_34_44[6, r]
        V[11, r] = V_34_44[7, r]
        V[16, r] = V_34_44[8, r]
        V[17, r] = V_34_44[9, r]
        V[18, r] = V_34_44[10, r]
        V[19, r] = V_34_44[11, r]
        V[24, r] = V_34_44[12, r]
        V[25, r] = V_34_44[13, r]
        V[26, r] = V_34_44[14, r]
        V[27, r] = V_34_44[15, r]
        W[0, r] = W_34_44[0, r]
        W[1, r] = W_34_44[1, r]
        W[2, r] = W_34_44[2, r]
        W[3, r] = W_34_44[3, r]
        W[4, r] = W_34_44[4, r]
        W[5, r] = W_34_44[5, r]
        W[6, r] = W_34_44[6, r]
        W[7, r] = W_34_44[7, r]
        W[8, r] = W_34_44[8, r]
        W[9, r] = W_34_44[9, r]
        W[10, r] = W_34_44[10, r]
        W[11, r] = W_34_44[11, r]
        
    for r in range(44):
        U[:, 44 + r] = U_34_44[:, r]
        V[4, 44 + r] = V_34_44[0, r]
        V[5, 44 + r] = V_34_44[1, r]
        V[6, 44 + r] = V_34_44[2, r]
        V[7, 44 + r] = V_34_44[3, r]
        V[12, 44 + r] = V_34_44[4, r]
        V[13, 44 + r] = V_34_44[5, r]
        V[14, 44 + r] = V_34_44[6, r]
        V[15, 44 + r] = V_34_44[7, r]
        V[20, 44 + r] = V_34_44[8, r]
        V[21, 44 + r] = V_34_44[9, r]
        V[22, 44 + r] = V_34_44[10, r]
        V[23, 44 + r] = V_34_44[11, r]
        V[28, 44 + r] = V_34_44[12, r]
        V[29, 44 + r] = V_34_44[13, r]
        V[30, 44 + r] = V_34_44[14, r]
        V[31, 44 + r] = V_34_44[15, r]
        W[12, 44 + r] = W_34_44[0, r]
        W[13, 44 + r] = W_34_44[1, r]
        W[14, 44 + r] = W_34_44[2, r]
        W[15, 44 + r] = W_34_44[3, r]
        W[16, 44 + r] = W_34_44[4, r]
        W[17, 44 + r] = W_34_44[5, r]
        W[18, 44 + r] = W_34_44[6, r]
        W[19, 44 + r] = W_34_44[7, r]
        W[20, 44 + r] = W_34_44[8, r]
        W[21, 44 + r] = W_34_44[9, r]
        W[22, 44 + r] = W_34_44[10, r]
        W[23, 44 + r] = W_34_44[11, r]
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    if time.time() < deadline - 10:
        try:
            U, V, W = _build_3x4_times_4x8()
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list):
                rank = len(U_list[0])
                if rank < best_rank:
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
        except Exception:
            pass
            
    if not _verify(best['U'], best['V'], best['W']):
        best = _schoolbook()
        
    return best
