import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define blocks
    # S0: indices 0, 1. Start 0.
    # S1: indices 2, 3. Start 2.
    # S2: index 4. Start 4.
    
    blocks = {
        0: (0, [0, 1]),
        1: (2, [2, 3]),
        2: (4, [4])
    }
    
    # Strassen blocks for I, J, K in {0, 1}
    for I in [0, 1]:
        for J in [0, 1]:
            for K in [0, 1]:
                start_I, idx_I = blocks[I]
                start_J, idx_J = blocks[J]
                start_K, idx_K = blocks[K]
                
                # Helper to get global indices
                def gA(a, b):
                    return (start_I + a) * m + (start_J + b)
                def gB(b, c):
                    return (start_J + b) * p + (start_K + c)
                def gC(c, a):
                    return (start_K + c) * n + (start_I + a)
                
                # 7 Strassen terms
                # Term 1: (A00 + A11)(B00 + B11)
                U[gA(0,0)][col] += 1
                U[gA(1,1)][col] += 1
                V[gB(0,0)][col] += 1
                V[gB(1,1)][col] += 1
                W[gC(0,0)][col] += 1
                W[gC(1,1)][col] += 1
                col += 1
                
                # Term 2: (A10 + A11)B00
                U[gA(1,0)][col] += 1
                U[gA(1,1)][col] += 1
                V[gB(0,0)][col] += 1
                W[gC(1,0)][col] += 1
                W[gC(1,1)][col] -= 1
                col += 1
                
                # Term 3: A00(B01 - B11)
                U[gA(0,0)][col] += 1
                V[gB(0,1)][col] += 1
                V[gB(1,1)][col] -= 1
                W[gC(0,1)][col] += 1
                W[gC(1,1)][col] += 1
                col += 1
                
                # Term 4: A11(B10 - B00)
                U[gA(1,1)][col] += 1
                V[gB(1,0)][col] += 1
                V[gB(0,0)][col] -= 1
                W[gC(0,0)][col] += 1
                W[gC(1,0)][col] += 1
                col += 1
                
                # Term 5: (A00 + A01)B11
                U[gA(0,0)][col] += 1
                U[gA(0,1)][col] += 1
                V[gB(1,1)][col] += 1
                W[gC(0,0)][col] -= 1
                W[gC(0,1)][col] += 1
                col += 1
                
                # Term 6: (A10 - A00)(B00 + B01)
                U[gA(1,0)][col] += 1
                U[gA(0,0)][col] -= 1
                V[gB(0,0)][col] += 1
                V[gB(0,1)][col] += 1
                W[gC(1,1)][col] += 1
                col += 1
                
                # Term 7: (A01 - A11)(B10 + B11)
                U[gA(0,1)][col] += 1
                U[gA(1,1)][col] -= 1
                V[gB(1,0)][col] += 1
                V[gB(1,1)][col] += 1
                W[gC(0,0)][col] += 1
                W[gC(1,1)][col] += 1
                col += 1
                
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by Strassen (i, j, k < 4)
                if i < 4 and j < 4 and k < 4:
                    continue
                U[i * m + j][col] += 1
                V[j * p + k][col] += 1
                W[k * n + i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
