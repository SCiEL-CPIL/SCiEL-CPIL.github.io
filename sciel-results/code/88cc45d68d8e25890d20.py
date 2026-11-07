"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # L1[i][j] = (i + j) % n
    # L2[i][j] = (i + 2*j) % n
    # These two are mutually orthogonal for n=10.
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Verify types are Python int
    # The modulo operation on Python ints returns Python ints.
    
    return [L1, L2]
# EVOLVE-BLOCK-END
