import time

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We stop at 85s.
    deadline = 85.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # We need to check n in [1000, 1000000]
    # The numbers involved in factorization are n+1 to n+24.
    # Max number is 1000000 + 24 = 1000024.
    max_val = 1000000 + 24
    
    # Precompute masks for each number up to max_val
    # masks[i] is a bitmask where bit j is set if primes[j] divides i
    masks = [0] * (max_val + 1)
    
    for i, p in enumerate(primes):
        bit = 1 << i
        # Mark multiples of p
        for j in range(p, max_val + 1, p):
            masks[j] |= bit
            
    # Counts of each prime in the current window
    prime_counts = [0] * num_primes
    current_distinct = 0
    
    def add_num(num):
        nonlocal current_distinct
        m = masks[num]
        # Iterate through the 9 primes
        for i in range(num_primes):
            if m & (1 << i):
                if prime_counts[i] == 0:
                    current_distinct += 1
                prime_counts[i] += 1
                
    def remove_num(num):
        nonlocal current_distinct
        m = masks[num]
        for i in range(num_primes):
            if m & (1 << i):
                prime_counts[i] -= 1
                if prime_counts[i] == 0:
                    current_distinct -= 1

    # Initialize the first window for n=1000
    # The window contains numbers 1001 to 1024
    for k in range(1, 25):
        add_num(1000 + k)
        
    best_distinct = current_distinct
    best_n = 1000
    
    # Slide the window
    # n ranges from 1000 to 1000000
    for n in range(1000, 1000000 + 1):
        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = n
            
        # Check time limit
        if time.time() > deadline:
            break
            
        # Slide: remove the number leaving the window (n+1)
        # and add the number entering the window (n+25)
        remove_num(n + 1)
        
        # The new number is n+25. 
        # For n=1000, new is 1025.
        # For n=999999, new is 1000024.
        # This is always <= max_val.
        add_num(n + 25)
            
    return best_n
