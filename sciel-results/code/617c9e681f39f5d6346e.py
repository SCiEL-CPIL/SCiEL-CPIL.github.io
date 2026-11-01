"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
"""
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the factor matrices U, V, W for the schoolbook decomposition.
    n=2, m=2, p=4.
    Rank = n*m*p = 16.
    U: (n*m, rank) = (4, 16)
    V: (m*p, rank) = (8, 16)
    W: (p*n, rank) = (8, 16)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Deadline guard: 150s limit, leave 5s margin
    start_time = time.time()
    deadline = start_time + 145.0
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time occasionally (though loop is tiny)
                if time.time() > deadline:
                    # Return whatever is filled so far, though for this small size it's fine
                    return {"U": U, "V": V, "W": W}
                
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
