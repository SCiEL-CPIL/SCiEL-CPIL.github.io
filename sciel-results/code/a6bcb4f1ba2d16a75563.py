import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    # Get all divisors of base > 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort ascending: Small moduli have fewer residue choices (rigid constraints), 
    # so they should be placed first. Large moduli are flexible and can fill gaps.
    moduli = sorted(divisors)
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in moduli:
        if time.time() - start_time > time_limit:
            break
            
        best_r = 0
        min_covered_count = base # Max possible covered
        
        for r in range(m):
            # Count already covered
            cnt = covered[r::m].sum()
            if cnt < min_covered_count:
                min_covered_count = cnt
                best_r = r
                
        classes.append([m, best_r])
        covered[best_r::m] = 1
        
    return classes
