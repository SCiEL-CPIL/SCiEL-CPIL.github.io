import time
import math
import random
import numpy as np

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

    primes = sieve(200)
    
    def tau_perp_fast(factors):
        # factors: list of (p, e)
        divs = [1]
        for p, e in factors:
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            new_divs = []
            for d in divs:
                for pow_val in powers:
                    new_divs.append(d * pow_val)
            divs = new_divs
        
        if len(divs) < 2:
            return 0
            
        arr = np.array(divs, dtype=np.int64)
        arr.sort()
        
        # np.gcd is vectorized
        gcds = np.gcd(arr[:-1], arr[1:])
        return int(np.sum(gcds == 1))

    best_n = 1
    best_score = 0
    best_factors = []
    
    # Initialize with square-free product of smallest primes
    # Product of first 15 primes is ~ 6e17
    init_primes = primes[:15]
    init_exps = [1] * 15
    init_factors = list(zip(init_primes, init_exps))
    
    prod = 1
    for p, e in init_factors:
        prod *= p ** e
        
    if prod > 10**18:
        # Remove largest primes until it fits
        while prod > 10**18 and len(init_factors) > 0:
            p, e = init_factors.pop()
            prod //= p ** e
        init_factors = [(p, 1) for p, e in init_factors]
        prod = 1
        for p, e in init_factors:
            prod *= p ** e
            
    if init_factors:
        s = tau_perp_fast(init_factors)
        if s > best_score:
            best_score = s
            best_n = prod
            best_factors = list(init_factors)
    else:
        # Fallback
        best_n = 2
        best_factors = [(2,1)]
        best_score = 1

    current_factors = list(best_factors)
    current_prod = best_n
    current_score = best_score

    improved = True
    no_improve_count = 0
    restart_count = 0
    max_restarts = 5

    while time.time() < deadline:
        if not improved:
            no_improve_count += 1
            if no_improve_count > 20 and restart_count < max_restarts:
                # Random restart: perturb current solution or start fresh
                # Simple perturbation: change a random exponent or prime
                if current_factors:
                    idx = random.randint(0, len(current_factors)-1)
                    p, e = current_factors[idx]
                    # Try a random new exponent
                    new_e = random.randint(1, 5)
                    base = current_prod // (p ** e)
                    new_prod = base * (p ** new_e)
                    if 2 <= new_prod <= 10**18:
                        current_factors[idx] = (p, new_e)
                        current_prod = new_prod
                        current_score = tau_perp_fast(current_factors)
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_prod
                            best_factors = list(current_factors)
                        improved = True
                        no_improve_count = 0
                        restart_count += 1
                    else:
                        # Try swapping a prime
                        q = random.choice(primes)
                        if q != p:
                            base = current_prod // (p ** e)
                            new_prod = base * (q ** e)
                            if 2 <= new_prod <= 10**18:
                                current_factors[idx] = (q, e)
                                current_prod = new_prod
                                current_score = tau_perp_fast(current_factors)
                                if current_score > best_score:
                                    best_score = current_score
                                    best_n = current_prod
                                    best_factors = list(current_factors)
                                improved = True
                                no_improve_count = 0
                                restart_count += 1
                else:
                    break
            elif no_improve_count > 100:
                break

        if time.time() > deadline - 5.0:
            break

        improved = False
        best_move_score = current_score
        best_new_factors = None
        best_new_prod = current_prod
        
        # 1. Exponent changes
        for i, (p, e) in enumerate(current_factors):
            # Decrease
            if e > 1:
                new_e = e - 1
                new_prod = current_prod // p
                if new_prod >= 2:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, new_e)
                    s = tau_perp_fast(new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod
                        
            # Increase
            if current_prod * p <= 10**18:
                new_e = e + 1
                new_prod = current_prod * p
                new_factors = list(current_factors)
                new_factors[i] = (p, new_e)
                s = tau_perp_fast(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 2. Swap primes
        current_primes_set = {p for p, e in current_factors}
        for i, (p, e) in enumerate(current_factors):
            for q in primes:
                if q in current_primes_set:
                    continue
                base = current_prod // (p ** e)
                new_prod = base * (q ** e)
                if new_prod > 10**18 or new_prod < 2:
                    continue
                new_factors = list(current_factors)
                new_factors[i] = (q, e)
                s = tau_perp_fast(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 3. Add new prime
        for q in primes:
            if q in current_primes_set:
                continue
            new_prod = current_prod * q
            if new_prod > 10**18:
                continue
            new_factors = list(current_factors)
            new_factors.append((q, 1))
            s = tau_perp_fast(new_factors)
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
            
            if current_score > best_score:
                best_score = current_score
                best_n = current_prod
                best_factors = list(current_factors)
        else:
            improved = False

    return best_n
