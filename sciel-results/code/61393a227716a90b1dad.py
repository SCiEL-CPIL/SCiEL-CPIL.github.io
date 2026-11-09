import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    Rank = n * m * p.
    """
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Time check (though construction is very fast)
                if time.time() > deadline:
                    # Return best found so far (which is complete if we haven't timed out mid-loop)
                    # Since we fill columns sequentially, if we time out, we have a partial decomposition?
                    # No, we need a valid decomposition. The schoolbook is built in one go.
                    # If we time out, we can't return a partial one.
                    # But construction takes < 0.1s, so this is safe.
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
