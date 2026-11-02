import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate initial solution
    # 2. ILS Loop
    # 3. Return best
