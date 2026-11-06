import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Step 1: Generate smallest primes p ≡ 1 (mod 4)
    # We need to find how many fit in 10^18
    limit = 10**18
    primes = []
    prod = 1
    candidate = 5
    while True:
        # Check if candidate is prime and candidate % 4 == 1
        if candidate % 4 == 1 and is_prime(candidate):
            if prod * candidate > limit:
                break
            primes.append(candidate)
            prod *= candidate
        candidate += 2
        
    if not primes:
        # Fallback to a known small case if something goes wrong
        # But 5 is always prime and 5 <= 10^18
        return {'N': 5, 'pairs': [[1, 4]]} # 1+4=5, 1=1^2, 4=2^2. Powerful.
        
    N = prod
    
    # Step 2: Find representation p = x^2 + y^2 for each prime
    # Use Cornacchia's algorithm or simple search for small primes
    # Since primes are small (first 11 are < 200), simple search is fine
    reps = []
    for p in primes:
        found = False
        x = 1
        while x * x < p:
            y2 = p - x * x
            y = math.isqrt(y2)
            if y * y == y2:
                reps.append((x, y))
                found = True
                break
            x += 1
        if not found:
            # Should not happen for p ≡ 1 mod 4
            reps.append((1, 0)) # Fallback
            
    # Step 3: Generate all pairs (u, v) such that u^2 + v^2 = N
    # Start with (1, 0)
    pairs = [(1, 0)]
    
    for x, y in reps:
        new_pairs = []
        for u, v in pairs:
            # (u + i v) * (x + i y) = (ux - vy) + i (uy + vx)
            u1 = u * x - v * y
            v1 = u * y + v * x
            # (u + i v) * (x - i y) = (ux + vy) + i (uy - vx)
            u2 = u * x + v * y
            v2 = u * y - v * x
            
            new_pairs.append((u1, v1))
            new_pairs.append((u2, v2))
        pairs = new_pairs
        
    # Step 4: Filter and format pairs
    # We need a <= b, so u <= v.
    # Also a, b > 0.
    # Distinct pairs.
    result_pairs = set()
    for u, v in pairs:
        # Ensure positive
        if u < 0: u = -u
        if v < 0: v = -v
        # Ensure u <= v
        if u > v:
            u, v = v, u
        # Skip if zero? 0 is not powerful? 
        # "1 counts as powerful". 0 is not positive integer usually. 
        # Task says "Integers ... a+b=N". Usually positive integers.
        # Powerful numbers are positive integers.
        if u == 0 or v == 0:
            continue
        # u^2 and v^2 are the components
        a = u * u
        b = v * v
        if a + b == N:
            result_pairs.add((a, b))
            
    # Convert to list of lists
    final_pairs = [list(p) for p in sorted(result_pairs)]
    
    return {'N': N, 'pairs': final_pairs}

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n == 3:
        return True
    if n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
