import time
import math
import numpy as np

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    if len(U) == 0:
        return 0.0
        
    n = len(U)
    if n == 1:
        return 1.0

    sumset = set()
    diffset = set()
    
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    ratio = len(diffset) / len(sumset)
    
    num = math.log(ratio, 2)
    den = math.log(2 * max_u + 1, 2)
    
    if den == 0:
        return 1.0
        
    return 1.0 + num / den

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    U = [0]
    best_score = compute_score(U)
    best_U = U
    U_set = set(U)
    
    limit = 10
    max_limit = 5000
    
    while time.time() < deadline - 2.0:
        improved = False
        best_x = None
        best_new_score = best_score
        
        # Try adding x from 1 to limit
        for x in range(1, limit + 1):
            if x in U_set:
                continue
                
            new_U = U + [x]
            score = compute_score(new_U)
            
            if score > best_new_score + 1e-12:
                best_new_score = score
                best_x = x
                improved = True
        
        if improved:
            U = U + [best_x]
            U_set.add(best_x)
            best_score = best_new_score
            best_U = U
            # Increase limit slowly
            limit = int(limit * 1.2) + 1
            if limit > max_limit:
                limit = max_limit
        else:
            # No improvement in current range
            # Expand range
            limit = int(limit * 1.5) + 1
            if limit > max_limit:
                # Try a few random large jumps
                found = False
                for _ in range(50):
                    x = max(U) + 1 + np.random.randint(0, 1000)
                    if x in U_set:
                        continue
                    new_U = U + [x]
                    score = compute_score(new_U)
                    if score > best_score + 1e-12:
                        U = new_U
                        U_set.add(x)
                        best_score = score
                        best_U = U
                        found = True
                        break
                if not found:
                    break
            
    return best_U
