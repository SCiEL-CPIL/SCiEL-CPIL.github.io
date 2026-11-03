import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_START_TIME = time.time()
TIME_BUDGET = 115

def _time_left():
    return TIME_BUDGET - (time.time() - _START_TIME)

def _verify(U, V, W, n=3, m=3, p=8):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

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

def _quick_search(best_rank, best_result):
    n, m, p = DIMENSIONS
    n_m, m_p, p_n = n * m, m * p, p * n
    
    # Try a few ranks with random init and 1-2 alternating steps
    for rank in range(min(best_rank - 1, 40), 30, -1):
        if _time_left() < 5:
            break
        for attempt in range(5):
            if _time_left() < 3:
                break
            U = np.random.randn(n_m, rank)
            V = np.random.randn(m_p, rank)
            W = np.random.randn(p_n, rank)
            for _ in range(3):
                if _time_left() < 3:
                    break
                for i in range(n):
                    for j in range(m):
                        A = np.array([[V[j*p+k, t] * W[k*n+i, t] for t in range(rank)] for k in range(p)])
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, t] * W[k*n+i, t] for t in range(rank)] for i in range(n)])
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i*m+j, t] * V[j*p+k, t] for t in range(rank)] for j in range(m)])
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return rank, {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return best_rank, best_result

def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_result = _schoolbook()
    if _time_left() > 10:
        new_rank, new_result = _quick_search(best_rank, best_result)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = new_result
    return best_result
