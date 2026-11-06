import math
import random
import time

def _tau_perp_from_factors(factors):
    """
    Calculate tau_perp(n) given the prime factorization of n.
    factors: dict {prime: exponent}
    """
    # Generate divisors
    divisors = [1]
    for p, e in factors.items():
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
        
    if len(divisors) < 2:
        return 0

    divisors.sort()
    
    score = 0
    prev = divisors[0]
    for i in range(1, len(divisors)):
        curr = divisors[i]
        if math.gcd(prev, curr) == 1:
            score += 1
        prev = curr
        
    return score

def _is_valid(factors, limit=10**18):
    """Check if the product of primes^exponents is <= limit."""
    n = 1
    for p, e in factors.items():
        n *= p ** e
        if n > limit:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Extended prime list
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Initial candidates
    # 1. Primorial of various sizes
    candidates = []
    for num_primes in [10, 12, 14, 15, 16]:
        if num_primes <= len(primes):
            cand = {p: 1 for p in primes[:num_primes]}
            if _is_valid(cand):
                candidates.append(cand)
    
    # 2. Add some structured candidates with higher exponents on small primes
    # 2^a * 3^b * 5^c * ...
    structured_candidates = [
        {2: 5, 3: 3, 5: 2, 7: 1, 11: 1, 13: 1},
        {2: 4, 3: 3, 5: 2, 7: 2, 11: 1, 13: 1, 17: 1},
        {2: 3, 3: 3, 5: 2, 7: 2, 11: 2, 13: 1, 17: 1},
        {2: 4, 3: 2, 5: 2, 7: 1, 11: 1, 13: 1, 17: 1, 19: 1},
        {2: 3, 3: 2, 5: 2, 7: 1, 11: 1, 13: 1, 17: 1, 19: 1, 23: 1},
    ]
    for cand in structured_candidates:
        if _is_valid(cand):
            candidates.append(cand)
    
    # 3. Random initial candidates
    for _ in range(20):
        factors = {}
        max_e = random.randint(2, 15)
        for i, p in enumerate(primes[:18]):
            if max_e == 0:
                break
            e = random.randint(0, max_e)
            if e > 0:
                factors[p] = e
                max_e = e
            else:
                break
        if factors and _is_valid(factors):
            candidates.append(factors)
    
    if not candidates:
        return 2
    
    best_factors = candidates[0]
    best_score = _tau_perp_from_factors(best_factors)
    
    # Local Search with simulated annealing
    current_factors = dict(best_factors)
    current_score = best_score
    temperature = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Mutation
        new_factors = dict(current_factors)
        
        # Get list of primes in current factors
        active_primes = [p for p in primes if p in new_factors]
        
        if not active_primes:
            current_factors = {primes[0]: 1}
            continue

        action = random.random()
        
        if action < 0.3:
            # Increase exponent of a random active prime
            p = random.choice(active_primes)
            new_factors[p] += 1
        elif action < 0.6:
            # Decrease exponent of a random active prime
            p = random.choice(active_primes)
            if new_factors[p] > 1:
                new_factors[p] -= 1
            else:
                del new_factors[p]
        elif action < 0.8:
            # Swap: decrease one, increase another
            if len(active_primes) >= 2:
                p1, p2 = random.sample(active_primes, 2)
                if new_factors[p1] > 1:
                    new_factors[p1] -= 1
                    new_factors[p2] = new_factors.get(p2, 0) + 1
                else:
                    # Try adding a new prime
                    used_primes = set(new_factors.keys())
                    available_primes = [p for p in primes if p not in used_primes]
                    if available_primes:
                        p = available_primes[0]
                        new_factors[p] = 1
                        del new_factors[p1]
            else:
                # Add next unused prime
                used_primes = set(new_factors.keys())
                available_primes = [p for p in primes if p not in used_primes]
                if available_primes:
                    p = available_primes[0]
                    new_factors[p] = 1
        else:
            # Try to add a new prime or remove one
            used_primes = set(new_factors.keys())
            available_primes = [p for p in primes if p not in used_primes]
            if available_primes and random.random() < 0.5:
                # Add smallest available
                p = available_primes[0]
                new_factors[p] = 1
            elif active_primes:
                # Remove a random active prime
                p = random.choice(active_primes)
                del new_factors[p]
        
        # Check validity
        if not _is_valid(new_factors):
            # Try to repair by reducing exponents of largest primes
            if new_factors:
                sorted_items = sorted(new_factors.items(), key=lambda x: x[0], reverse=True)
                for p, e in sorted_items:
                    if e > 1:
                        new_factors[p] -= 1
                        if _is_valid(new_factors):
                            break
                if not _is_valid(new_factors):
                    continue
            else:
                continue
        
        # Normalize: sort primes, sort exponents in descending order, and pair them
        if new_factors:
            sorted_primes = sorted(new_factors.keys())
            sorted_exponents = sorted(new_factors.values(), reverse=True)
            normalized = {}
            for p, e in zip(sorted_primes, sorted_exponents):
                if e > 0:
                    normalized[p] = e
            new_factors = normalized
        
        # Re-check validity after normalization
        if not _is_valid(new_factors):
            continue
        
        score = _tau_perp_from_factors(new_factors)
        
        if score > best_score:
            best_score = score
            best_factors = new_factors
            current_factors = new_factors
            current_score = score
        else:
            # Accept with probability based on temperature (simulated annealing)
            delta = score - current_score
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_factors = new_factors
                current_score = score
            
            # Occasionally restart from best
            if random.random() < 0.02:
                current_factors = dict(best_factors)
                current_score = best_score
        
        # Cool down
        temperature *= cooling_rate
        temperature = max(temperature, 0.1)
    
    # Convert best_factors to integer
    n = 1
    for p, e in best_factors.items():
        n *= p ** e
        
    return n
