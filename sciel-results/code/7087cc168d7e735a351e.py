import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # First try rank 10, fall back to rank 11
    # Known rank-11 solution
    rank11_U = None
    rank11_V = None
    rank11_W = None
    
    # ... (build rank 11 as before)
    
    # Try rank 10 search
    # ...
