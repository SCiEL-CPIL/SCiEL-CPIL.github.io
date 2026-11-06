import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    
    # Get all divisors > 1 of base
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    divisors.sort()
    
    # Greedy Initialization
    count = np.zeros(base, dtype=np.int32)
    classes = []
    
    for m in divisors:
        best_r = 0
        best_gain = -1
        for r in range(m):
            if time.time() - start_time > 110:
                break
            gain = np.sum(count[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        count[best_r::m] += 1
        
    # Local Search
    improved = True
    while improved:
        if time.time() - start_time > 115:
            break
        improved = False
        for i in range(len(classes)):
            m, r = classes[i]
            # Try all other residues
            for r_new in range(m):
                if r_new == r:
                    continue
                if time.time() - start_time > 118:
                    break
                
                # Calculate lost and gained using slicing
                lost = np.sum(count[r::m] == 1)
                gained = np.sum(count[r_new::m] == 0)
                
                if gained > lost:
                    # Apply move
                    count[r::m] -= 1
                    count[r_new::m] += 1
                    classes[i][1] = r_new
                    improved = True
                    # Restart local search
                    break
            if improved:
                break # Restart outer loop
        
    return classes
