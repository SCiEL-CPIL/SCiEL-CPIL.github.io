import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 57
    # Split m=5 into 2 and 3
    # Part 1: 3x2x5 (Rank 30)
    # Part 2: 3x3x5 -> Split p=5 into 2 and 3
    # Part 2a: 3x3x2 (Rank 18)
    # Part 2b: 3x3x3 (Rank 9)
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 3x2x5
    # A indices: i in 0..2, j in 0..1
    # B indices: j in 0..1, k in 0..4
    # C indices: k in 0..4, i in 0..2
    for i in range(3):
        for j in range(2):
            for k in range(5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col is now 30

    # Part 2a: 3x3x2
    # A indices: i in 0..2, j in 2..4 (local j 0..2)
    # B indices: j in 2..4 (local j 0..2), k in 0..1
    # C indices: k in 0..1, i in 0..2
    for i in range(3):
        for j_local in range(3):
            j_global = j_local + 2
            for k in range(2):
                U[i * 5 + j_global][col] = 1
                V[j_global * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col is now 30 + 18 = 48

    # Part 2b: 3x3x3
    # A indices: i in 0..2, j in 2..4 (local j 0..2)
    # B indices: j in 2..4 (local j 0..2), k in 2..4
    # C indices: k in 2..4, i in 0..2
    for i in range(3):
        for j_local in range(3):
            j_global = j_local + 2
            for k_local in range(3):
                k_global = k_local + 2
                U[i * 5 + j_global][col] = 1
                V[j_global * 5 + k_global][col] = 1
                W[k_global * 3 + i][col] = 1
                col += 1
    # col is now 48 + 9 = 57

    return {"U": U, "V": V, "W": W}
