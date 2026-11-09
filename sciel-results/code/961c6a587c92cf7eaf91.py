"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we didn't finish, we might have an incomplete decomposition.
    # But schoolbook is fast, so it should finish.
    # To be safe, if incomplete, we return a valid subset or restart?
    # Given the small size, it will finish.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
