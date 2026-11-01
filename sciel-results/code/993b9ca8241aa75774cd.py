<<<<<<< SEARCH
def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication (Strassen's algorithm).
    
    U shape: (4, 7) for A entries
    V shape: (4, 7) for B entries
    W shape: (4, 7) for C entries (index k*n+i)
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
=======
import numpy as np

def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication (Strassen's algorithm).
    
    U shape: (4, 7) for A entries
    V shape: (4, 7) for B entries
    W shape: (4, 7) for C entries (index k*n+i)
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices as numpy arrays
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
>>>>>>> REPLACE
