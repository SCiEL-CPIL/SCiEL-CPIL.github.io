import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    n = 1
    
    # Calculate LCM of 1..K
    # We can iterate and update LCM
    # LCM(a, b) = a * b // GCD(a, b)
    
    k = 1
    while True:
        next_n = math.lcm(n, k+1)
        if next_n > limit:
            break
        n = next_n
        k += 1
        if time.time() > deadline:
            return n
            
    return n
