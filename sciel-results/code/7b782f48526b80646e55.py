import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a certain limit to allow swaps.
    # Max prime factor of n <= 10^18 is 10^18, but we only need small primes for optimization.
    # Primes up to 10000 should be sufficient for local improvements.
    primes = sieve(10000)
    
    def get_divisors(factors):
        # factors: list of (p, e)
        divs = [1]
        for p, e in factors:
            new_divs = []
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            for d in divs:
                for pow_val in powers:
                    new_divs.append(d * pow_val)
            divs = new_divs
        return divs

    def tau_perp(n, factors):
        # factors: list of (p, e)
        if not factors:
            return 0
        divs = get_divisors(factors)
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_n = 1
    best_score = 0
    best_factors = []

    def update_best(n, factors, score):
        nonlocal best_n, best_score, best_factors
        if score > best_score:
            best_score = score
            best_n = n
            best_factors = list(factors)

    # Initial candidates
    # 1. First 15 primes
    init_primes = primes[:15]
    init_factors = [(p, 1) for p in init_primes]
    prod = 1
    for p, e in init_factors:
        prod *= p ** e
    if prod <= 10**18:
        s = tau_perp(prod, init_factors)
        update_best(prod, init_factors, s)

    # 2. First 14 primes, try adding next few
    # We will just let the search do this, but multiple starts help.
    
    # Local Search
    # State: factors (list of tuples), prod (int)
    
    def local_search(init_factors, init_prod):
        current_factors = list(init_factors)
        current_prod = init_prod
        current_score = tau_perp(current_prod, current_factors)
        update_best(current_prod, current_factors, current_score)
        
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            best_move_score = current_score
            best_new_factors = None
            best_new_prod = current_prod
            
            # Generate neighbors
            # 1. Swap prime
            current_primes_set = {p for p, e in current_factors}
            for i, (p, e) in enumerate(current_factors):
                # Try swapping p with q
                # q should be such that new_prod <= 10^18
                # base = current_prod // p**e
                base = current_prod // (p ** e)
                # We can try a range of q
                # If e=1, q can be up to 10^18 // base
                # But we only have primes up to 10000.
                # If we need larger q, we can't easily find them without a big sieve.
                # However, usually optimal solutions use small primes.
                for q in primes:
                    if q in current_primes_set:
                        continue
                    new_prod = base * (q ** e)
                    if new_prod > 10**18 or new_prod < 2:
                        continue
                    new_factors = list(current_factors)
                    new_factors[i] = (q, e)
                    s = tau_perp(new_prod, new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod

            # 2. Add prime
            for q in primes:
                if q in current_primes_set:
                    continue
                new_prod = current_prod * q
                if new_prod > 10**18:
                    continue
                new_factors = list(current_factors)
                new_factors.append((q, 1))
                s = tau_perp(new_prod, new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

            # 3. Remove prime (if exp > 1, reduce; if exp == 1, remove)
            for i, (p, e) in enumerate(current_factors):
                if e > 1:
                    new_e = e - 1
                    new_prod = current_prod // p
                    new_factors = list(current_factors)
                    new_factors[i] = (p, new_e)
                    s = tau_perp(new_prod, new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod
                else: # e == 1
                    if len(current_factors) > 1: # Don't remove if only 1 prime left? n>=2
                        new_factors = list(current_factors)
                        new_factors.pop(i)
                        new_prod = current_prod // p
                        s = tau_perp(new_prod, new_factors)
                        if s > best_move_score:
                            best_move_score = s
                            best_new_factors = new_factors
                            best_new_prod = new_prod

            # 4. Increase exponent
            for i, (p, e) in enumerate(current_factors):
                if current_prod * p <= 10**18:
                    new_e = e + 1
                    new_prod = current_prod * p
                    new_factors = list(current_factors)
                    new_factors[i] = (p, new_e)
                    s = tau_perp(new_prod, new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod

            if best_new_factors is not None:
                current_factors = best_new_factors
                current_prod = best_new_prod
                current_score = best_move_score
                improved = True
                no_improve_count = 0
                update_best(current_prod, current_factors, current_score)
            else:
                no_improve_count += 1
                if no_improve_count > 10:
                    break

    # Run local search from a few different starting points
    # Start 1: Primorial 15
    if time.time() < deadline:
        local_search([(p, 1) for p in primes[:15]], math.prod(primes[:15]))

    # Start 2: Primorial 14 (in case 15 is a local trap, though unlikely)
    if time.time() < deadline:
        p14 = primes[:14]
        prod14 = math.prod(p14)
        local_search([(p, 1) for p in p14], prod14)

    # Start 3: Random valid number
    if time.time() < deadline:
        # Try a random set of small primes
        random.seed(42)
        rand_primes = random.sample(primes[:50], 12)
        rand_primes.sort()
        prod = math.prod(rand_primes)
        if prod <= 10**18:
            local_search([(p, 1) for p in rand_primes], prod)

    return best_n
