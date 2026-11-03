import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    N = q ** n
    
    # Precomputation
    # ... (same as current) ...
    
    # Greedy Init
    # ... (same as current, using bitsets for speed) ...
    
    # Local Search
    # 1. Remove Redundant
    # 2. Swap Loop
