"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication tensor.
    
    Rank 7 is the known optimal (Strassen) for 2x2 matrix multiplication.
    The theoretical lower bound is 7, so no rank-6 decomposition exists.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    r = 7
    
    # Build U (4x7): row i*m+j corresponds to A[i][j]
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    # Build V (4x7): row j*p+k corresponds to B[j][k]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # Build W (4x7): row k*n+i corresponds to C[i][k] (trace convention)
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
