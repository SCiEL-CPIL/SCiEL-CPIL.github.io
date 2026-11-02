import time

DIMENSIONS = [2, 2, 16]
_start_time = time.time()
_TIMEOUT = 140.0

def _time_remaining():
    return _TIMEOUT - (time.time() - _start_time)

def construct():
    if _time_remaining() < 5:
        # Emergency fallback: return a valid rank-56 decomposition quickly
        n, m, p = DIMENSIONS
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for q in range(8):
            base_col = q * 7
            col = base_col
            U[0][col] = 1; U[3][col] = 1
            V[2*q][col] = 1; V[16+2*q+1][col] = 1
            W[4*q][col] = 1; W[4*q+3][col] = 1
            col = base_col + 1
            U[2][col] = 1; U[3][col] = 1
            V[2*q][col] = 1
            W[4*q+1][col] = 1; W[4*q+3][col] = -1
            col = base_col + 2
            U[0][col] = 1
            V[2*q+1][col] = 1; V[16+2*q+1][col] = -1
            W[4*q+2][col] = 1; W[4*q+3][col] = 1
            col = base_col + 3
            U[3][col] = 1
            V[16+2*q][col] = 1; V[2*q][col] = -1
            W[4*q][col] = 1; W[4*q+1][col] = 1
            col = base_col + 4
            U[0][col] = 1; U[1][col] = 1
            V[16+2*q+1][col] = 1
            W[4*q][col] = -1; W[4*q+2][col] = 1
            col = base_col + 5
            U[2][col] = 1; U[0][col] = -1
            V[2*q][col] = 1; V[2*q+1][col] = 1
            W[4*q+3][col] = 1
            col = base_col + 6
            U[1][col] = 1; U[3][col] = -1
            V[16+2*q][col] = 1; V[16+2*q+1][col] = 1
            W[4*q][col] = 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Predefine the 7 Strassen terms for a 2x2 block
    # Each term is a list of (index, value) for U, V, W
    # U indices: 0: A00, 1: A01, 2: A10, 3: A11
    # V indices: 0-15: B0,k, 16-31: B1,k
    # W indices: 0-31: C_i,k -> 2k+i
    
    # We will iterate q from 0 to 7
    # For each q, we add 7 columns
    
    # Term definitions relative to q
    # t=0: M1
    # t=1: M2
    # t=2: M3
    # t=3: M4
    # t=4: M5
    # t=5: M6
    # t=6: M7
    
    for q in range(8):
        base_col = q * 7
        
        # M1
        col = base_col
        U[0][col] = 1
        U[3][col] = 1
        V[2*q][col] = 1
        V[16 + 2*q + 1][col] = 1
        W[4*q][col] = 1
        W[4*q + 3][col] = 1
        
        # M2
        col = base_col + 1
        U[2][col] = 1
        U[3][col] = 1
        V[2*q][col] = 1
        W[4*q + 1][col] = 1
        W[4*q + 3][col] = -1
        
        # M3
        col = base_col + 2
        U[0][col] = 1
        V[2*q + 1][col] = 1
        V[16 + 2*q + 1][col] = -1
        W[4*q + 2][col] = 1
        W[4*q + 3][col] = 1
        
        # M4
        col = base_col + 3
        U[3][col] = 1
        V[16 + 2*q][col] = 1
        V[2*q][col] = -1
        W[4*q][col] = 1
        W[4*q + 1][col] = 1
        
        # M5
        col = base_col + 4
        U[0][col] = 1
        U[1][col] = 1
        V[16 + 2*q + 1][col] = 1
        W[4*q][col] = -1
        W[4*q + 2][col] = 1
        
        # M6
        col = base_col + 5
        U[2][col] = 1
        U[0][col] = -1
        V[2*q][col] = 1
        V[2*q + 1][col] = 1
        W[4*q + 3][col] = 1
        
        # M7
        col = base_col + 6
        U[1][col] = 1
        U[3][col] = -1
        V[16 + 2*q][col] = 1
        V[16 + 2*q + 1][col] = 1
        W[4*q][col] = 1

    return {"U": U, "V": V, "W": W}
