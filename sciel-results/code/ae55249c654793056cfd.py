import time

DIMENSIONS = [2, 4, 8]
_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def construct():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    """
    if _elapsed() > _DEADLINE:
        return _fallback_construct()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(2):
        if _elapsed() > _DEADLINE:
            break
        for j in range(4):
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # U
            U[a00][col] += 1
            U[a11][col] += 1
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            U[a00][col+2] += 1
            U[a11][col+3] += 1
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            
            # V
            V[b00][col] += 1
            V[b11][col] += 1
            V[b00][col+1] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            V[b11][col+4] += 1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            
            # W
            W[c00][col] += 1
            W[c00][col+3] += 1
            W[c00][col+4] += -1
            W[c00][col+6] += 1
            W[c01][col+2] += 1
            W[c01][col+4] += 1
            W[c10][col+1] += 1
            W[c10][col+3] += 1
            W[c11][col] += 1
            W[c11][col+1] += -1
            W[c11][col+2] += 1
            W[c11][col+5] += 1
            
            col += 7
    
    return {"U": U, "V": V, "W": W}

def _fallback_construct():
    """Simple schoolbook rank-64 decomposition as fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for k in range(p):
            for s in range(m):
                U[i * m + s][r] = 1
                V[s * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
