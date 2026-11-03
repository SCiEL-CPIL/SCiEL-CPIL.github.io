<<<<<<< SEARCH
"""Program for 3x5x6 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Construct a rank-84 decomposition of the 3x5x6 matrix multiplication tensor.
    Uses Strassen's rank-7 algorithm for 2x2x2 sub-blocks where applicable.
    """
    import time
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
=======
"""Program for 3x5x6 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Construct a rank-84 decomposition of the 3x5x6 matrix multiplication tensor.
    Uses Strassen's rank-7 algorithm for 2x2x2 sub-blocks where applicable.
    """
    import time
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
>>>>>>> REPLACE

