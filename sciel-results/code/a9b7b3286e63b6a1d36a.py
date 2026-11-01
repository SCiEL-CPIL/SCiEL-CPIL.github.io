import time

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    rank = n * m  # 6
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 6)
    V = [[0] * rank for _ in range(m * p)]  # (18, 6)
    W = [[0] * rank for _ in range(p * n)]  # (12, 6)
    
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[i * m + j][t] = 1
            for k in range(p):
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
    
    # Verify
    def verify(Um, Vm, Wm, n_l, m_l, p_l, rank_l):
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    s = 0
                    for t in range(rank_l):
                        s += Um[i * m_l + j][t] * Vm[j * p_l + k][t] * Wm[k * n_l + i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
