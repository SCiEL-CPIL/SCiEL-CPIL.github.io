import random
from itertools import combinations
from math import gcd
import time

def construct():
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    # Precompute GCDs for 1..128
    N = 128
    gcds = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    def is_valid_addition(selected, x):
        # Check if adding x to selected creates a bad triple
        # Bad triple: a, b in selected such that gcd(a,b) == gcd(a,x) == gcd(b,x)
        # Note: selected is a list of ints
        for i in range(len(selected)):
            a = selected[i]
            ga_x = gcds[a][x]
            for j in range(i+1, len(selected)):
                b = selected[j]
                if gcds[a][b] == ga_x == gcds[b][x]:
                    return False
        return True

    best_set = []
    
    # Run multiple random restarts
    for _ in range(1000):
        if time.time() - start_time > timeout:
            break
            
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        current = []
        
        for x in candidates:
            if is_valid_addition(current, x):
                current.append(x)
                
        if len(current) > len(best_set):
            best_set = current
            
    return best_set
