import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Splits
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    # Calculate total rank
    total_rank = 0
    # Pre-calculate block ranks
    # ...
    
    # Allocate
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    # Iterate blocks
    # ...
    
    return {"U": U, "V": V, "W": W}
