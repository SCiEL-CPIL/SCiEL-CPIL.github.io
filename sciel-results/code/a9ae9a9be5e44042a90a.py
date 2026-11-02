"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, but numpy might be faster for init?
    # The task allows NumPy. Let's use numpy for cleaner construction, then convert to lists if needed?
    # The return interface expects {'U', 'V', 'W'}. The seed returns lists of lists.
    # I will stick to lists of lists to be safe, or check if numpy arrays are accepted.
    # "factor matrices of shapes... with integer/half-integer entries"
    # Usually, lists are safer if the evaluator expects specific types, but numpy arrays are often fine.
    # The seed uses lists. I will use lists.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    # Return current best (which is complete since it's static)
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
