import numpy as np
import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s for safety
    
    # Try to find a smaller tournament
    # We try sizes that are multiples of something? Or just descending.
    # Known lower bound 31.
    # We'll try 60, 62, 64, 66.
    # If we find one, return it.
    
    for n in [66, 64, 62, 60, 58, 56, 54, 52, 50]:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5.0:
            break
        result = try_find_tournament(n, remaining)
        if result is not None:
            return result
            
    # Fallback to Paley 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            orientation.append(0 if (j - i) % p in squares else 1)
    return {'n': p, 'orientation': orientation}

def try_find_tournament(n, time_budget):
    if n < 5:
        return None
        
    # Initialize random tournament
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            val = random.randint(0, 1)
            A[i, j] = val
            A[j, i] = 1 - val
            
    # Helper to check if a set is bad
    def is_bad(a, b, c, d):
        # a,b,c,d are distinct indices
        # Check if any dominates the other 3
        # a dominates: A[a,b]==1, A[a,c]==1, A[a,d]==1
        if A[a,b] == 1 and A[a,c] == 1 and A[a,d] == 1:
            return False
        if A[b,a] == 1 and A[b,c] == 1 and A[b,d] == 1:
            return False
        if A[c,a] == 1 and A[c,b] == 1 and A[c,d] == 1:
            return False
        if A[d,a] == 1 and A[d,b] == 1 and A[d,c] == 1:
            return False
        return True

    # Initial count
    bad_count = 0
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    if is_bad(i, j, k, l):
                        bad_count += 1
                        
    if bad_count == 0:
        return format_result(n, A)

    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    start = time.time()
    
    # Precompute list of other vertices for each u,v? 
    # No, just loop.
    
    while (time.time() - start) < time_budget:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Calculate old bad count for affected sets
        old_bad = 0
        # Affected sets contain u and v
        # Iterate x, y
        # To avoid duplicate {u,v,x,y} and {u,v,y,x}, assume x < y
        # But we must exclude u,v
        
        # Optimization: 
        # The set is {u,v,x,y}.
        # We can iterate x from 0 to n-1, y from x+1 to n-1.
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if is_bad(u, v, x, y):
                    old_bad += 1
                    
        # Flip
        A[u, v] = 1 - A[u, v]
        A[v, u] = 1 - A[v, u]
        
        new_bad = 0
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if is_bad(u, v, x, y):
                    new_bad += 1
                    
        delta_bad = new_bad - old_bad
        
        if delta_bad < 0 or (delta_bad > 0 and random.random() < np.exp(-delta_bad / T)):
            bad_count += delta_bad
            if bad_count == 0:
                return format_result(n, A)
        else:
            # Revert
            A[u, v] = 1 - A[u, v]
            A[v, u] = 1 - A[v, u]
            
        T *= cooling
        if T < min_T:
            break
            
    return None

def format_result(n, A):
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            orientation.append(int(A[i, j]))
    return {'n': n, 'orientation': orientation}
