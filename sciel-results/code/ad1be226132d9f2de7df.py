import math

def construct():
    limit = 10**18
    
    # 1. Generate initial primorial
    primes = []
    candidate = 2
    n = 1
    while n * candidate <= limit:
        # Check if candidate is prime
        if is_prime(candidate):
            n *= candidate
            primes.append(candidate)
        candidate += 1 if candidate < 2 else 2 # simple increment, is_prime handles it

    # Actually, let's use a more standard prime generation for the initial set
    primes = []
    n = 1
    c = 2
    while n * c <= limit:
        if is_prime(c):
            n *= c
            primes.append(c)
        c += 1 if c < 2 else 2

    # 2. Define score function
    def get_score(num, factors):
        # factors is a list of (prime, exponent)
        # Generate divisors
        divs = [1]
        for p, e in factors:
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Greedy optimization
    # Current factors: all exponents 1
    current_factors = [(p, 1) for p in primes]
    current_n = n
    current_score = get_score(current_n, current_factors)
    
    # Try to increase exponents greedily
    # We iterate through primes in ascending order
    for i, p in enumerate(primes):
        # Current exponent is current_factors[i][1]
        while current_n * p <= limit:
            # Calculate new score
            # Update factors
            new_factors = current_factors.copy()
            new_factors[i] = (p, new_factors[i][1] + 1)
            
            trial_n = current_n * p
            trial_score = get_score(trial_n, new_factors)
            
            if trial_score > current_score:
                current_n = trial_n
                current_score = trial_score
                current_factors = new_factors
            else:
                break

    return current_n

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
