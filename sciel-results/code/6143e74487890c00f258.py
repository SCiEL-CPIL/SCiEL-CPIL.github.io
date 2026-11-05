import time

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Returns the schoolbook decomposition for 3x3x8 matrix multiplication.
    Rank = 3 * 3 * 8 = 72.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    assert len(res['U']) == 9
    assert len(res['U'][0]) == 72
    assert len(res['V']) == 24
    assert len(res['V'][0]) == 72
    assert len(res['W']) == 24
    assert len(res['W'][0]) == 72
