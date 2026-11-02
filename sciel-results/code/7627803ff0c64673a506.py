"""Combined program for the suite task; uses schoolbook decomposition with robustness checks."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    # Safety margin: The task allows 120s, we aim to finish in < 1s, 
    # but keep a guard to prevent any unexpected infinite loops or hangs.
    deadline = time.time() + 115
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists for compatibility
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time occasionally to satisfy the "deadline guard" directive
                if time.time() > deadline:
                    # Return whatever we have (which is complete if we didn't hang)
                    # In this specific loop, it's very fast, so this is just a formality
                    return {"U": U, "V": V, "W": W}
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
