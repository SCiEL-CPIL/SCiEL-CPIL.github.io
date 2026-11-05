"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 3x3x7 matrix multiplication.
    Rank is 63.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps (i, j) to column
                U[i * m + j][column] = 1.0
                # V maps (j, k) to column
                V[j * p + k][column] = 1.0
                # W maps (k, i) to column (Note: Task says W[k*n+i, r])
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# Guard against timeout if this were part of a search loop, though construct() is O(1)
if __name__ == "__main__":
    start = time.time()
    result = construct()
    # Basic sanity check
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n
    assert all(len(row) == rank for row in result['U'])
    assert all(len(row) == rank for row in result['V'])
    assert all(len(row) == rank for row in result['W'])
    elapsed = time.time() - start
    print(f"Schoolbook Rank: {rank}, Time: {elapsed:.4f}s")
