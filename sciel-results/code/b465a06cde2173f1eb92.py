import time
import numpy as np
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Helper to check if adding x to set S (with counts C) is valid
    # S is a list, C is a numpy array
    def can_add(x, S, C):
        # Create a copy of C to check
        # We only need to check the sums involving x
        # Sums are x+y for y in S, and 2x
        # We can check if any of these sums already have count 2
        # If count is 2, adding x makes it 3 -> invalid
        # If count is 1, adding x makes it 2 -> valid
        # If count is 0, adding x makes it 1 -> valid
        
        # Optimization: just check the specific sums
        for y in S:
            if C[x+y] >= 2:
                return False
        if C[2*x] >= 2:
            return False
        return True

    def run_greedy(order):
        S = []
        C = np.zeros(2001, dtype=np.int8)
        for x in order:
            if time.time() - start_time > time_limit:
                return S
            if can_add(x, S, C):
                S.append(x)
                for y in S:
                    C[x+y] += 1
                C[2*x] += 1
        return S

    # 1. Standard Greedy
    S1 = run_greedy(range(1, 1001))
    best_set = S1
    best_size = len(S1)
    
    # 2. Randomized Greedy
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        order = list(range(1, 1001))
        random.shuffle(order)
        S = run_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # 3. Local Search (optional, if time permits)
    # Try to add more elements to best_set
    # This is complex, so we might skip for now or do a simple pass
    
    return best_set
