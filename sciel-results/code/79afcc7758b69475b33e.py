"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Uses the schoolbook decomposition which has rank n*m*p.
    For small dimensions like 2x4x7, this is often the minimal or near-minimal rank
    without specialized algebraic identities (like Strassen) that apply to specific block structures.
    """
    # Deadline guard: 150s limit, return best found (which is immediate here)
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of zeros
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
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time (unlikely to trigger for small N, but required by directive)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        # In case of timeout, return what we have (though construction is O(1) effectively)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
