def construct():
    from math import gcd
    
    N = 512
    
    # Precompute lcm table for speed
    # lcm[i][j] for i, j in 1..N
    # Using a dictionary or 2D list. 512x512 is ~260k ints.
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            l = i * j // gcd(i, j)
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
    
    # Try all M in 1..N as base
    for M in range(1, N + 1):
        base = get_divisors(M)
        # Sort base to ensure deterministic greedy
        base.sort()
        
        # Quick check: if base is already invalid (shouldn't happen for divisors of M)
        # But we assume divisors of M are valid.
        
        current = set(base)
        
        # Greedy expansion
        # Try to add elements from 1 to N not in current
        # We can try ascending order
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
            
    # If we found a set, return it
    if not best_set:
        return [1]
        
    return best_set
