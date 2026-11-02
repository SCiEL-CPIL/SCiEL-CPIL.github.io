import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Fallback for general case (seed logic)
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 3**n
    powers = [3**i for i in range(n)]
    
    # Precompute neighbors
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        temp = i
        digits = [0]*n
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        
        for j in range(n):
            orig = digits[j]
            for new_val in range(3):
                if new_val != orig:
                    new_i = i + (new_val - orig) * powers[j]
                    neighbors[i].append(new_i)
    
    covered = np.zeros(N, dtype=bool)
    uncovered_list = list(range(N))
    code = []
    
    while uncovered_list:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_gain = 0
        
        for u in uncovered_list:
            gain = 0
            for v in neighbors[u]:
                if not covered[v]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = u
                if best_gain == 17:
                    break
        
        if best_idx == -1:
            break
            
        code.append(best_idx)
        
        to_remove = set()
        to_remove.add(best_idx)
        if not covered[best_idx]:
            covered[best_idx] = True
        
        for v in neighbors[best_idx]:
            if not covered[v]:
                covered[v] = True
                to_remove.add(v)
                
        if to_remove:
            # Filter uncovered_list
            # Optimization: if to_remove is small, we could use a set for uncovered_list?
            # But list is fine for 6561.
            uncovered_list = [x for x in uncovered_list if x not in to_remove]
            
    # Convert to lists
    result = []
    for idx in code:
        temp = idx
        digits = [0]*n
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        result.append(digits)
        
    return result
