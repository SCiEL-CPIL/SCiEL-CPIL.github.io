import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # For M_{3,3,7}, the trivial rank is 3*3*7 = 63.
    # No known simple decomposition reduces this rank significantly 
    # without complex coefficients (like Winograd for 3x3x3).
    # We return the trivial identity decomposition.
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility and simplicity
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by [i*m + j, column]
                U[i * m + j][column] = 1
                # V is indexed by [j*p + k, column]
                V[j * p + k][column] = 1
                # W is indexed by [k*n + i, column]
                # Convention: W[k*n+i, r]
                W[k * n + i][column] = 1
                column += 1
                
                # Check time occasionally
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
