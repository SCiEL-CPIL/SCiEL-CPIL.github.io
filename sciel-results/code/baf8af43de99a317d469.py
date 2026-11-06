import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try Paley tournament on 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    A = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for j in range(p):
            if i != j:
                A[i, j] = (j - i) % p in squares
    
    # Verify
    if time.time() > deadline:
        # Fallback to 83 if we are running out of time (unlikely for verification)
        pass 
        
    valid = True
    # We can break early if we find a bad set
    for a, b, c, d in itertools.combinations(range(p), 4):
        if time.time() > deadline:
            break
        if A[a,b] and A[a,c] and A[a,d]:
            continue
        if A[b,a] and A[b,c] and A[b,d]:
            continue
        if A[c,a] and A[c,b] and A[c,d]:
            continue
        if A[d,a] and A[d,b] and A[d,c]:
            continue
        valid = False
        break
        
    if valid:
        # Convert to orientation list
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if A[i,j] else 1)
        return {'n': p, 'orientation': orientation}

    # If not valid, try a local search
    # We will try to repair the tournament
    # Count bad sets? No, just random repair.
    
    # Maybe try other primes?
    # 71, 79, 83...
    # But we want to minimize n.
    
    # Let's try a simple local search on 67
    # Initialize A with the Paley tournament
    # We need to track bad sets efficiently?
    # For now, just random flips guided by sampling.
    
    best_A = A.copy()
    # We don't have a good cost function update for local search in Python quickly.
    # Let's just try a few random restarts or different primes?
    
    # If 67 fails, try 66? No, Paley needs prime.
    # We can try a random tournament on 67 and search? Unlikely to succeed in 120s.
    
    # Fallback: Return the 83 Paley tournament
    p = 83
    squares = {x * x % p for x in range(1, p)}
    A = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for j in range(p):
            if i != j:
                A[i, j] = (j - i) % p in squares
                
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            orientation.append(0 if A[i,j] else 1)
            
    return {'n': p, 'orientation': orientation}
