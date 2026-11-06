import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute LCM for all pairs to speed up checks
    # lcm[i][j] for i, j in 1..N
    # Using a 2D list. 513x513 is ~263k ints.
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            l = i * j // g
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs
    
    best_set = []
    best_size = 0
    
    # Strategy: Seed with divisors of M, then greedily expand.
    # Iterate through all M in 1..N
    for M in range(1, N + 1):
        if time.time() > deadline:
            break
            
        base = get_divisors(M)
        base.sort()
        current = set(base)
        
        # Greedy expansion
        # Try to add elements from 1 to N not in current
        # We try ascending order, which tends to keep numbers small and compatible
        for x in range(1, N + 1):
            if x in current:
                continue
            
            # Check if x is compatible with all in current
            valid = True
            for y in current:
                if lcm_table[x][y] > N:
                    valid = False
                    break
            
            if valid:
                current.add(x)
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
            
    # Fallback if no set found (should not happen)
    if not best_set:
        return list(range(1, 23))
        
    return best_set
