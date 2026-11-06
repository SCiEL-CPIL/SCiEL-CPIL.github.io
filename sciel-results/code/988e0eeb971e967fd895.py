"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    
    # Step 1: Generate primes p ≡ 1 (mod 4)
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes_1mod4 = []
    candidate = 5
    while len(primes_1mod4) < 20:
        if candidate % 4 == 1 and is_prime(candidate):
            primes_1mod4.append(candidate)
        candidate += 2

    # Step 2: Find exponents to maximize product (e_i + 1)
    # We use a simple backtracking/greedy approach for the first few primes
    # The optimal exponents are non-increasing.
    
    best_score = 0
    best_exponents = []
    best_N = 1
    
    # Max exponent for the smallest prime 5 is around 25
    max_e = 25
    
    def backtrack(idx, current_N, current_score, exponents):
        nonlocal best_score, best_exponents, best_N
        if idx == len(primes_1mod4):
            if current_score > best_score:
                best_score = current_score
                best_exponents = exponents[:]
                best_N = current_N
            return
        
        p = primes_1mod4[idx]
        # Determine max exponent for this prime given current_N
        # p^e <= 10^18 / current_N
        if current_N > 10**18:
            return
            
        max_possible_e = 0
        temp = current_N
        for e in range(1, 30):
            if temp * p > 10**18:
                break
            temp *= p
            max_possible_e = e
            
        # Also constrained by previous exponent if idx > 0
        if idx > 0:
            max_possible_e = min(max_possible_e, exponents[idx-1])
        
        # Try exponents from max_possible_e down to 1
        # We can also try 0, but since we are building from smallest primes,
        # skipping a prime might allow higher exponents on larger primes?
        # Usually, for max product, we use all small primes.
        for e in range(max_possible_e, 0, -1):
            new_N = current_N * (p ** e)
            if new_N > 10**18:
                continue
            exponents.append(e)
            backtrack(idx + 1, new_N, current_score * (e + 1), exponents)
            exponents.pop()

    backtrack(0, 1, 1, [])
    
    N = best_N
    exponents = best_exponents
    
    # Step 3: Find a^2 + b^2 = p for each prime p
    # For p = a^2 + b^2
    def find_sq_sum(p):
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a*a
            b = int(math.isqrt(b2))
            if b*b == b2:
                return (a, b)
        return (1, 1) # Fallback, shouldn't happen for p=1 mod 4

    prime_factors = []
    for i, p in enumerate(primes_1mod4):
        if i < len(exponents) and exponents[i] > 0:
            a, b = find_sq_sum(p)
            prime_factors.append((p, a, b, exponents[i]))
    
    # Step 4: Generate all Gaussian integer representations
    # z = prod (a + ib)^f (a - ib)^(e-f)
    # We represent Gaussian integers as tuples (x, y)
    
    def multiply_gauss(z1, z2):
        x1, y1 = z1
        x2, y2 = z2
        return (x1*x2 - y1*y2, x1*y2 + y1*x2)
    
    def power_gauss(z, k):
        result = (1, 0)
        base = z
        while k > 0:
            if k % 2 == 1:
                result = multiply_gauss(result, base)
            base = multiply_gauss(base, base)
            k //= 2
        return result
    
    # Start with 1
    current_gauss = [(1, 0)]
    
    for p, a, b, e in prime_factors:
        z1 = (a, b)
        z2 = (a, -b)
        next_gauss = []
        for z in current_gauss:
            for f in range(e + 1):
                # z * (z1)^f * (z2)^(e-f)
                part1 = power_gauss(z1, f)
                part2 = power_gauss(z2, e - f)
                prod = multiply_gauss(part1, part2)
                new_z = multiply_gauss(z, prod)
                next_gauss.append(new_z)
        current_gauss = next_gauss
        
    # Step 5: Convert to pairs (x^2, y^2)
    pairs = set()
    for x, y in current_gauss:
        # x^2 + y^2 = N
        # We want a = x^2, b = y^2 with a <= b
        a = x*x
        b = y*y
        if a > b:
            a, b = b, a
        # Ensure a, b are positive? 
        # If x=0 or y=0, one is 0. Powerful numbers usually >= 1.
        # The task says 1 counts as powerful. 0 is not a positive integer usually.
        # N >= 2. If x=0, a=0. Is 0 powerful? "Integers... 1 counts". 
        # Usually powerful numbers are positive integers.
        # If a=0, it might be invalid. Let's check constraints.
        # "a and b are powerful... 1 counts as powerful". 
        # If a=0, 0 is not in the set of powerful numbers defined by prime factors.
        # So we should skip pairs where a=0 or b=0.
        if a == 0 or b == 0:
            continue
        pairs.add((a, b))
        
    # Convert to list of lists
    pair_list = [list(p) for p in pairs]
    
    return {'N': N, 'pairs': pair_list}
# EVOLVE-BLOCK-END
