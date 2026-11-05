"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # 1. Generate primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    # We need enough primes. Product of first 15 primes is ~6e17.
    # Product of first 16 is ~3e19, so max distinct primes is 15.
    # However, we might use larger primes with exponent 1 if we drop small ones?
    # Actually, to maximize divisor count and coprime pairs, small primes are generally better
    # because they allow for higher exponents or more combinations within the limit.
    # Let's use first 20 primes for the search space, but most will likely be 0.
    primes = sieve(100)
    max_primes = 16
    
    # 2. Helper to calculate score efficiently
    # We need to generate divisors, sort them, and count coprime adjacent pairs.
    # Optimizing divisor generation:
    # Instead of list comprehensions which create many intermediate lists,
    # we can use a more direct approach, but for n <= 10^18, the number of divisors
    # is at most ~100k (max for 10^18 is 103680 for n=897612484786617600).
    # 100k divisors sort is fast (~0.01s). GCD check is fast.
    
    def calc_score(exps):
        # Reconstruct n
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes[i] ** e
        if n > 10**18 or n < 2:
            return -1, n
        
        # Get divisors
        divs = [1]
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        
        divs.sort()
        
        # Calculate score
        score = 0
        # Optimization: gcd(a, b) == 1
        # For small numbers, this is very fast.
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score, n

    # 3. Initialization
    # Start with a strong seed: Product of first 15 primes (squarefree)
    # This is a good local optimum.
    init_exps = [0] * max_primes
    for i in range(15):
        init_exps[i] = 1
        
    best_exps = init_exps[:]
    best_score, _ = calc_score(best_exps)
    
    # 4. Simulated Annealing
    # Temperature starts high, cools down.
    # Move set:
    # - Increase exponent of random prime by 1
    # - Decrease exponent of random prime by 1 (if > 0)
    # - Swap exponents of two primes? (Rare, but might help)
    # - Jump: Set exponent to random value in [0, max_possible]
    
    current_exps = best_exps[:]
    current_score, _ = calc_score(current_exps)
    
    T = 10.0
    T_min = 0.1
    alpha = 0.995  # Cooling rate
    
    # We need to estimate the cost of one iteration.
    # calc_score is the bottleneck.
    # Let's measure time for one iteration.
    t0 = time.time()
    for _ in range(10):
        # Dummy move
        pass
    # Actually, let's just run until deadline.
    
    steps = 0
    while time.time() < deadline:
        if T < T_min:
            # Restart with slightly higher T to escape deep local minima
            # or just stop if we've been stable for a long time?
            # Let's try a few restarts.
            T = 5.0
            # Perturb current state
            for _ in range(5):
                i = random.randint(0, max_primes - 1)
                if current_exps[i] > 0:
                    current_exps[i] -= 1
                else:
                    current_exps[i] += 1
            current_score, _ = calc_score(current_exps)
            if current_score > best_score:
                best_score = current_score
                best_exps = current_exps[:]
            
        # Generate a neighbor
        move_type = random.random()
        new_exps = current_exps[:]
        
        if move_type < 0.4:
            # Increase exponent
            i = random.randint(0, max_primes - 1)
            # Check if we can increase without exceeding limit
            # Rough check: n * p_i <= 10^18
            # We'll just generate and let calc_score reject if too big
            new_exps[i] += 1
        elif move_type < 0.8:
            # Decrease exponent
            i = random.randint(0, max_primes - 1)
            if new_exps[i] > 0:
                new_exps[i] -= 1
            else:
                # If 0, maybe increase instead? Or pick another prime
                new_exps[i] += 1 
        else:
            # Random jump in exponent
            i = random.randint(0, max_primes - 1)
            # Max exponent for p_i is floor(log_p(10^18))
            max_e = int(math.log(10**18, primes[i])) if primes[i] > 1 else 0
            new_exps[i] = random.randint(0, max_e)
            
        new_score, n_val = calc_score(new_exps)
        
        if new_score < 0:
            # Invalid n, reject
            continue
            
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_exps = new_exps
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_exps = current_exps[:]
                
        T *= alpha
        steps += 1
        
        # Safety check
        if time.time() > deadline - 5:
            break
            
    # Final n
    final_n = 1
    for i in range(max_primes):
        if best_exps[i] > 0:
            final_n *= primes[i] ** best_exps[i]
            
    if final_n > 10**18 or final_n < 2:
        # Fallback to a known valid number
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
        
    return final_n
# EVOLVE-BLOCK-END
