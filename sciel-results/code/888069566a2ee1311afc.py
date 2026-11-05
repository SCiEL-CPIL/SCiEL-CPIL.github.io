import time

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 512
    best_M = 1
    max_divs = 0
    
    # Find M <= 512 with maximum number of divisors
    for M in range(1, N + 1):
        # Count divisors of M
        count = 0
        i = 1
        while i * i <= M:
            if M % i == 0:
                count += 1
                if i != M // i:
                    count += 1
            i += 1
        
        if count > max_divs:
            max_divs = count
            best_M = M
            
        # Check time
        if time.time() - start_time > DEADLINE:
            break
    
    # Generate divisors of best_M
    divisors = []
    i = 1
    while i * i <= best_M:
        if best_M % i == 0:
            divisors.append(i)
            if i != best_M // i:
                divisors.append(best_M // i)
        i += 1
    
    # Sort for consistency, though not strictly required by "subset"
    divisors.sort()
    
    # Verify validity (just in case, though logic guarantees it)
    # For a, b in divisors, lcm(a,b) divides best_M <= 512.
    
    return divisors
