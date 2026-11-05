"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Find an integer n <= 10^18 that maximizes tau_perp(n), the number of
    adjacent coprime pairs in the sorted divisor list.
    
    Strategy: Use n = p1^a1 * p2^a2 * ... * pk^ak where p_i are distinct primes.
    The number of divisors is tau(n) = (a1+1)(a2+1)...(ak+1).
    The divisors can be seen as a multidimensional grid. Adjacent coprime pairs
    are relatively rare. We want to maximize the number of adjacent pairs (in sorted order)
    that are coprime.
    
    Empirically, for a given number of prime factors k, using small primes with
    high exponents increases the density of divisors, which might help or hurt
    the adjacency of coprime pairs depending on the structure.
    
    However, a known approach for maximizing the count of *something* related to
    coprime adjacencies in divisor lists often involves primorials or numbers
    with many small prime factors.
    
    Let's use a search to find the best n.
    Given the 120s limit, we can afford a significant search space.
    
    Key insight: The score is the number of adjacent pairs d_i, d_{i+1} with gcd(d_i, d_{i+1}) = 1.
    This is a non-trivial global property of the sorted divisor list.
    
    Let's try to search over exponent vectors for the first few primes.
    Primes: 2, 3, 5, 7, 11, 13, ...
    The product of the first 15 primes is ~6.14e17.
    n = 2^a2 * 3^a3 * ...
    
    We can perform a randomized local search or a beam search.
    Since we need to compute the divisors and sort them to get the score, and
    for n ~ 10^18 with many divisors (e.g., 2^30 ~ 10^9 divisors is too many to list explicitly),
    we must be careful.
    
    Wait, if n is close to 10^18, and has many divisors, listing them might be expensive.
    Max divisors for n < 10^18 is around 10^5? 
    Let's check. 2^a * 3^b * 5^c...
    If n = 2^a * 3^b, divisors ~ (a+1)(b+1).
    To get (a+1)(b+1) ~ 10^5, with 2^a * 3^b ~ 10^18,
    a log 2 + b log 3 = 18 log 10 => 0.3 a + 0.477 b = 4.14 * 10 (roughly, log10)
    Actually ln(n) = 18 * ln(10) ~ 41.4
    a ln 2 + b ln 3 = 41.4
    Maximize (a+1)(b+1) subject to a ln 2 + b ln 3 = 41.4.
    Using Lagrange multipliers: 1/(a+1) / ln 2 = 1/(b+1) / ln 3
    (b+1) ln 3 = (a+1) ln 2 => b+1 = (a+1) ln 2 / ln 3.
    Substitute into constraint: a ln 2 + ((a+1) ln 2 / ln 3 - 1) ln 3 = 41.4
    a ln 2 + (a+1) ln 2 - ln 3 = 41.4
    (2a+1) ln 2 - ln 3 = 41.4
    2a ln 2 = 41.4 + 1.1 - 0.69 = 41.8
    a = 41.8 / 1.386 = 30.1.
    So a=30, b+1 = 31 * 0.366 = 11.3 => b=10.
    n = 2^30 * 3^10 = 10^9 * 59049 ~ 6e13. This is well below 10^18.
    Divisors: 31 * 11 = 341.
    
    We can include more primes to get closer to 10^18 and more divisors.
    If we use primes 2..43 (14 primes), exponents all 1:
    Product ~ 1.3e16. Divisors 2^14 = 16384.
    This is very manageable to list and sort.
    
    Let's try to maximize divisors closer to 10^18.
    Candidate: n = 2^6 * 3^4 * 5^3 * 7^2 * 11 * 13 * 17 * 19 * 23 * 29 * 31 ...
    Let's just do a search.
    
    The evaluation of tau_perp(n) requires generating divisors, sorting, and checking gcd.
    If tau(n) is small (< 100,000), this is fast.
    
    We will search for n of the form product p_i^a_i where a_i are small integers.
    We want to maximize the score.
    
    Let's define a function to compute the score.
    """
    
    start_time = time.time()
    deadline = 110  # seconds
    
    # Generate first few primes
    def generate_primes(limit_count):
        primes = []
        candidate = 2
        while len(primes) < limit_count:
            is_p = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(candidate)
            candidate += 1
        return primes

    primes = generate_primes(20)  # 2, 3, ..., 71
    
    # Helper to compute tau_perp(n)
    def calc_score(n):
        if n < 2:
            return 0
        
        # Factorize n
        # Since n is constructed from known primes, we could store the factors.
        # But for general robustness in the search, let's factorize the candidate n.
        # Actually, in the search we build n from primes, so we know the factors.
        # But let's write a generic factorizer for the check, or pass factors.
        
        # Let's assume n is composed of primes from our list.
        # We will build n and its factorization in the search.
        # For now, let's write a function that takes n and factors it.
        
        factors = {}
        temp_n = n
        for p in primes:
            if p * p > temp_n:
                break
            while temp_n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp_n //= p
        if temp_n > 1:
            factors[temp_n] = factors.get(temp_n, 0) + 1
            
        # Generate divisors
        divs = [1]
        for p, exp in factors.items():
            current_divs = []
            for d in divs:
                for e in range(exp + 1):
                    current_divs.append(d * (p ** e))
            divs = current_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
                
        return score, len(divs)
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # Initial guess
    
    # Search strategy:
    # The exponents should be non-increasing for optimal divisor count (Hardy-Ramanujan intuition).
    # But for tau_perp, we don't know the exact structure.
    # Let's try a randomized approach with local improvements.
    
    # Randomized search with restarts
    import random
    
    # Precompute prime powers
    max_exp = 20
    prime_powers = {}
    for i, p in enumerate(primes):
        prime_powers[i] = [1]
        val = 1
        for e in range(1, max_exp + 1):
            val *= p
            if val > 10**18:
                break
            prime_powers[i].append(val)
            
    # We restrict to the first 10-15 primes to keep search space manageable
    # but allow exponents.
    
    num_primes_to_use = 12
    search_primes = primes[:num_primes_to_use]
    
    # Initialize with a good candidate
    # Primorial of first 15 primes is ~ 6.1e17
    # Let's start with exponents [1,1,1,1,1,1,1,1,1,1,1,1]
    
    current_exp = [1] * num_primes_to_use
    current_n = 1
    for i in range(num_primes_to_use):
        current_n *= search_primes[i]
        
    current_score, _ = calc_score(current_n)
    best_score = max(best_score, current_score)
    if current_score > best_score:
        best_n = current_n
        
    # Local search / simulated annealing-like
    # Randomly pick an exponent to change
    
    # To speed up, we can skip recalculation if n doesn't change or is invalid
    
    while time.time() < deadline:
        # Pick a random prime index to modify
        i = random.randint(0, num_primes_to_use - 1)
        # Pick a random change: increase, decrease, or reset
        r = random.random()
        if r < 0.3:
            # Increase
            new_exp = current_exp[i] + 1
        elif r < 0.6:
            # Decrease
            new_exp = current_exp[i] - 1
        else:
            # Random small value
            new_exp = random.randint(0, 5)
            
        if new_exp < 0:
            continue
            
        # Check if product exceeds limit
        old_val = current_n
        old_p_power = search_primes[i] ** current_exp[i]
        new_p_power = search_primes[i] ** new_exp
        
        if old_p_power == 0:
            continue # shouldn't happen
            
        new_n = (old_n // old_p_power) * new_p_power
        
        if new_n > 10**18:
            continue
        if new_n < 2:
            continue
            
        # Evaluate
        score, _ = calc_score(new_n)
        
        # Accept if better, or with some probability (annealing not implemented for simplicity, greedy mostly)
        # Greedy local search can get stuck.
        # Let's try a simple random hill climbing with restarts.
        if score > best_score:
            best_score = score
            best_n = new_n
            current_exp[i] = new_exp
            current_n = new_n
        else:
            # With some probability, accept worse solutions to escape local minima
            # Simple temperature cooling?
            # For now, just accept if random < 0.1 to explore
            if random.random() < 0.1:
                current_exp[i] = new_exp
                current_n = new_n
                
        # Occasionally restart?
        # If stuck for a long time, reset exponents to random?
        
    return best_n
# EVOLVE-BLOCK-END
