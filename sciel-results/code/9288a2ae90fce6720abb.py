import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # The optimal strategy derived is to include 2 and all odd numbers in [129, 256]
    # Let's verify if any other small number combination works.
    # We established that any two small numbers a, b <= 16 have lcm(a,b) <= 256.
    # So we can pick at most one small number.
    # We compared 2, 3, 4, 5, 8, 16, 32, 64, 128.
    # 2 gave the best delta.
    
    # Construct the set
    selected = [2]
    for y in range(129, 257):
        if y % 2 != 0:
            selected.append(y)
            
    # Verify validity just in case
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            if math.lcm(selected[i], selected[j]) <= 256:
                # Fallback to base set if error (should not happen)
                return list(range(129, 257))
                
    if time.time() > deadline:
        return selected
        
    return selected
