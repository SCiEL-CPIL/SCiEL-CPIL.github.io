import time

def construct():
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    LIMIT = 10000
    counts = [0] * (LIMIT + 1)
    current_L = 0
    residues = []
    
    # S0: indices x where counts[x] == 0
    # S1: indices x where counts[x] == 1
    S0 = set(range(1, LIMIT + 1))
    S1 = set()
    
    for p in primes:
        tally1 = [0] * p
        tally0 = [0] * p
        
        # Primary goal: fix single-covered positions
        for x in S1:
            r = x % p
            tally1[r] += 1
            
        # Secondary goal: cover zero-covered positions
        for x in S0:
            r = x % p
            tally0[r] += 1
        
        # Find best residue
        best_r = 0
        max_t1 = -1
        max_t0 = -1
        for r in range(p):
            if (tally1[r], tally0[r]) > (max_t1, max_t0):
                max_t1 = tally1[r]
                max_t0 = tally0[r]
                best_r = r
        
        residues.append(best_r)
        
        # Apply best_r
        start = best_r if best_r != 0 else p
        
        if start <= LIMIT:
            for x in range(start, LIMIT + 1, p):
                old_c = counts[x]
                new_c = old_c + 1
                counts[x] = new_c
                
                if old_c == 0:
                    S0.remove(x)
                    S1.add(x)
                elif old_c == 1:
                    S1.remove(x)
                # else: old_c >= 2, no change to sets
        
        # Update current_L
        while current_L < LIMIT and counts[current_L + 1] >= 2:
            current_L += 1
            
    return residues
