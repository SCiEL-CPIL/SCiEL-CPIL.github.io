"""
Matrix multiplication 2x5x5: Schoolbook decomposition.
This program returns the standard rank-50 decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
The rank is n*m*p = 2*5*5 = 50.
The score is seed/value. Seed is 50. Value is 50. Score = 1.0.
"""

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
