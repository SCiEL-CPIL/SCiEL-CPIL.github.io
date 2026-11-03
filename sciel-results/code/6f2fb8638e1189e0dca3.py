"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    Rank = n * m * p.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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
                # U maps input A indices (i, j) to column
                U[i * m + j][column] = 1
                # V maps input B indices (j, k) to column
                V[j * p + k][column] = 1
                # W maps output C indices (k, i) to column
                # Note: Output index is k*n + i according to task description W[k*n+i, r]
                W[k * n + i][column] = 1
                column += 1
                
                # Check deadline (though this loop is very fast)
                if time.time() > deadline:
                    # If we somehow ran out of time (unlikely for schoolbook), return what we have
                    # But since we are building it, we should just finish.
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
