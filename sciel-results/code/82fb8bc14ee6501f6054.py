import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for half in range(2):
        a_off = half * 4
        b_off = half * 10
        
        # Strassen for cols 0,1
        # U indices: a_off, a_off+1, a_off+2, a_off+3
        # V indices: b_off, b_off+1, b_off+5, b_off+6
        # W indices: 0, 1, 2, 3
        
        # M1
        U[a_off][col] += 1; U[a_off+3][col] += 1
        V[b_off][col] += 1; V[b_off+6][col] += 1
        W[0][col] += 1; W[3][col] += 1
        col += 1
        # M2
        U[a_off+2][col] += 1; U[a_off+3][col] += 1
        V[b_off][col] += 1
        W[2][col] += 1
        col += 1
        # M3
        U[a_off][col] += 1
        V[b_off+1][col] += 1; V[b_off+5][col] -= 1
        W[1][col] += 1
        col += 1
        # M4
        U[a_off+3][col] += 1
        V[b_off][col] += 1; V[b_off+5][col] -= 1
        W[0][col] += 1
        col += 1
        # M5
        U[a_off][col] += 1; U[a_off+1][col] += 1
        V[b_off+5][col] += 1
        W[0][col] -= 1; W[1][col] += 1
        col += 1
        # M6
        U[a_off+2][col] += 1; U[a_off][col] -= 1
        V[b_off][col] += 1; V[b_off+5][col] += 1
        W[2][col] += 1
        col += 1
        # M7
        U[a_off+1][col] += 1; U[a_off+3][col] -= 1
        V[b_off+1][col] += 1; V[b_off+5][col] += 1
        W[3][col] += 1; W[0][col] -= 1
        col += 1
        
        # Strassen for cols 2,3
        # V offset +2, W offset +4
        # M1
        U[a_off][col] += 1; U[a_off+3][col] += 1
        V[b_off+2][col] += 1; V[b_off+8][col] += 1
        W[4][col] += 1; W[7][col] += 1
        col += 1
        # M2
        U[a_off+2][col] += 1; U[a_off+3][col] += 1
        V[b_off+2][col] += 1
        W[6][col] += 1
        col += 1
        # M3
        U[a_off][col] += 1
        V[b_off+3][col] += 1; V[b_off+7][col] -= 1
        W[5][col] += 1
        col += 1
        # M4
        U[a_off+3][col] += 1
        V[b_off+2][col] += 1; V[b_off+7][col] -= 1
        W[4][col] += 1
        col += 1
        # M5
        U[a_off][col] += 1; U[a_off+1][col] += 1
        V[b_off+7][col] += 1
        W[4][col] -= 1; W[5][col] += 1
        col += 1
        # M6
        U[a_off+2][col] += 1; U[a_off][col] -= 1
        V[b_off+2][col] += 1; V[b_off+7][col] += 1
        W[6][col] += 1
        col += 1
        # M7
        U[a_off+1][col] += 1; U[a_off+3][col] -= 1
        V[b_off+3][col] += 1; V[b_off+7][col] += 1
        W[7][col] += 1; W[4][col] -= 1
        col += 1
        
        # Schoolbook for col 4
        U[a_off][col] = 1; V[b_off+4][col] = 1; W[8][col] = 1
        col += 1
        U[a_off+1][col] = 1; V[b_off+9][col] = 1; W[8][col] = 1
        col += 1
        U[a_off+2][col] = 1; V[b_off+4][col] = 1; W[9][col] = 1
        col += 1
        U[a_off+3][col] = 1; V[b_off+9][col] = 1; W[9][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
