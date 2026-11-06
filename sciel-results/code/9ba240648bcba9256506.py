import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Step 1: Generate primes congruent to 1 mod 4
    # We need primes up to roughly 1000 to get 12-13 factors
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes_1_mod_4 = []
    candidate = 5
    while len(primes_1_mod_4) < 20:
        if is_prime(candidate) and candidate % 4 == 1:
            primes_1_mod_4.append(candidate)
        candidate += 2
    
    # Step 2: Construct N as product of smallest primes
    N = 1
    k = 0
    for p in primes_1_mod_4:
        if N * p > 10**18:
            break
        N *= p
        k += 1
    
    if k < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Step 3: Find all representations N = x^2 + y^2
    # Using the recursive identity:
    # (a^2+b^2)(c^2+d^2) = (ac-bd)^2 + (ad+bc)^2
    # Base case: p = x^2 + y^2. For p=5, (1,2).
    # We need to find the base representations for each prime factor.
    
    def find_rep_prime(p):
        # Find x, y such that x^2 + y^2 = p
        # Since p is prime and p % 4 == 1, a solution exists.
        # Simple search for small p
        for x in range(1, int(math.isqrt(p)) + 1):
            y2 = p - x*x
            y = int(math.isqrt(y2))
            if y*y == y2:
                return (x, y)
        return (1, 0) # Fallback, shouldn't happen for valid primes
    
    reps = []
    for p in primes_1_mod_4[:k]:
        reps.append(find_rep_prime(p))
    
    # Combine representations
    # Start with (1, 0) representing 1
    curr_x, curr_y = 1, 0
    for rx, ry in reps:
        # (curr_x^2 + curr_y^2) * (rx^2 + ry^2)
        # = (curr_x*rx - curr_y*ry)^2 + (curr_x*ry + curr_y*rx)^2
        new_x = abs(curr_x * rx - curr_y * ry)
        new_y = curr_x * ry + curr_y * rx
        # Ensure x <= y for canonical form? 
        # The identity generates one specific representation.
        # Actually, for each prime factor, there are two choices of sign/order in the combination?
        # No, the number of representations doubles with each prime.
        # The recursive formula above generates ONE representation.
        # To get ALL representations, we need to branch.
        
        # Let's restart with a proper recursive generation of all pairs.
        pass

    # Corrected Step 3: Generate all pairs (x,y) such that x^2+y^2=N
    # We can do this by maintaining a list of all (x,y) pairs for the product so far.
    # Start with [(1, 0)]
    # For each prime p = a^2 + b^2:
    #   For each existing (x, y):
    #     new1 = (x*a - y*b, x*b + y*a)
    #     new2 = (x*a + y*b, x*b - y*a) -> this is the same up to signs/swaps?
    #     Actually, (x^2+y^2)(a^2+b^2) = (xa-yb)^2 + (xb+ya)^2
    #     Also = (xa+yb)^2 + (xb-ya)^2
    #     These are the two distinct representations (up to symmetry).
    
    pairs_xy = [(1, 0)]
    for p in primes_1_mod_4[:k]:
        a, b = find_rep_prime(p)
        new_pairs = []
        for x, y in pairs_xy:
            # Branch 1
            nx1 = x * a - y * b
            ny1 = x * b + y * a
            new_pairs.append((nx1, ny1))
            
            # Branch 2
            nx2 = x * a + y * b
            ny2 = x * b - y * a
            new_pairs.append((nx2, ny2))
        
        pairs_xy = new_pairs
        
        if time.time() > deadline:
            break

    # Step 4: Convert (x,y) to powerful pairs (x^2, y^2)
    # Filter for x^2 + y^2 = N and x <= y
    # Note: The generated pairs might have negative coordinates or x > y.
    # We take absolute values and sort.
    
    final_powerful_pairs = []
    seen = set()
    
    for x, y in pairs_xy:
        if x*x + y*y != N:
            continue
        a = abs(x)
        b = abs(y)
        if a > b:
            a, b = b, a
        # Check if this pair was already added
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        final_powerful_pairs.append([a*a, b*b])
        
    if not final_powerful_pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': final_powerful_pairs}
