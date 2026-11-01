import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(k_a, k_b, start_col):
        # M1
        U[0][start_col] = 1
        U[3][start_col] = 1
        V[k_a][start_col] = 1
        V[11 + k_b][start_col] = 1
        W[2 * k_a][start_col] = 1
        W[2 * k_b + 1][start_col] = 1
        start_col += 1
        
        # M2
        U[2][start_col] = 1
        U[3][start_col] = 1
        V[k_a][start_col] = 1
        W[2 * k_a + 1][start_col] = 1
        start_col += 1
        
        # M3
        U[0][start_col] = 1
        V[k_b][start_col] = 1
        V[11 + k_b][start_col] = -1
        W[2 * k_b][start_col] = 1
        start_col += 1
        
        # M4
        U[3][start_col] = 1
        V[11 + k_a][start_col] = 1
        V[k_a][start_col] = -1
        W[2 * k_a + 1][start_col] = 1
        W[2 * k_a][start_col] = 1
        start_col += 1
        
        # M5
        U[0][start_col] = 1
        U[1][start_col] = 1
        V[11 + k_b][start_col] = 1
        W[2 * k_a][start_col] = -1
        W[2 * k_b][start_col] = 1
        start_col += 1
        
        # M6
        U[2][start_col] = 1
        U[0][start_col] = -1
        V[k_a][start_col] = 1
        V[k_b][start_col] = 1
        W[2 * k_b + 1][start_col] = 1
        start_col += 1
        
        # M7
        U[1][start_col] = 1
        U[3][start_col] = -1
        V[11 + k_a][start_col] = 1
        V[11 + k_b][start_col] = 1
        W[2 * k_a][start_col] = 1
        W[2 * k_b + 1][start_col] = 1
        
        return start_col
    
    for i in range(5):
        k_a = 2 * i
        k_b = 2 * i + 1
        col = add_strassen_block(k_a, k_b, col)
        
    # Schoolbook for k=10
    U[0][col] = 1
    V[10][col] = 1
    W[20][col] = 1
    col += 1
    
    U[1][col] = 1
    V[21][col] = 1
    W[20][col] = 1
    col += 1
    
    U[2][col] = 1
    V[10][col] = 1
    W[21][col] = 1
    col += 1
    
    U[3][col] = 1
    V[21][col] = 1
    W[21][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
