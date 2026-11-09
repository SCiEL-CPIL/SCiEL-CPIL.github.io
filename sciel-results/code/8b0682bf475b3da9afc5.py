import math
import time
import random

MAX_N = 10**18
DEADLINE = 145.0

# Generate primes up to 2000 to allow swapping in larger primes if needed
def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = generate_primes(2000)
PRIME_SET = set(PRIMES)

def get_divisors(exponents):
    """
    exponents: dict {prime: exponent}
    Returns sorted list of divisors
    """
    divs = [1]
    for p, e in exponents.items():
        if e == 0:
            continue
        new_divs = []
        # Precompute powers
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        for d in divs:
            for pow_p in powers:
                new_divs.append(d * pow_p)
        divs = new_divs
    divs.sort()
    return divs

def score_from_divs(divs):
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate(exponents):
    """
    exponents: dict {prime: exponent}
    Returns (score, n)
    """
    n = 1
    for p, e in exponents.items():
        if e > 0:
            n *= (p ** e)
            if n > MAX_N:
                return -1, n
    
    if n < 2:
        return -1, n
        
    divs = get_divisors(exponents)
    s = score_from_divs(divs)
    return s, n

def construct():
    start_time = time.time()
    
    # Initial best: First 15 primes, exponent 1
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    # Product is ~ 6.1e17
    init_exponents = {p: 1 for p in PRIMES[:15]}
    best_score, best_n = evaluate(init_exponents)
    best_exponents = init_exponents.copy()
    
    # Local Search
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        current_exponents = best_exponents.copy()
        current_score = best_score
        
        # Calculate current n
        current_n = 1
        for p, e in current_exponents.items():
            if e > 0:
                current_n *= (p ** e)
                
        # Generate candidate moves - prioritize by expected benefit
        candidates = []
        
        # 1. Increase exponent (high priority - adds structure)
        for p, e in current_exponents.items():
            if e > 0:
                if current_n * p <= MAX_N:
                    new_exp = current_exponents.copy()
                    new_exp[p] = e + 1
                    candidates.append((new_exp, 0))
                    
        # 2. Add new prime (high priority - adds coprime dimension)
        max_q = MAX_N // current_n
        used_primes = set(p for p, e in current_exponents.items() if e > 0)
        valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
        for q in valid_qs:
            new_exp = current_exponents.copy()
            new_exp[q] = 1
            candidates.append((new_exp, 1))
        
        # 3. Decrease exponent (lower priority)
        for p, e in current_exponents.items():
            if e > 1:
                new_exp = current_exponents.copy()
                new_exp[p] = e - 1
                candidates.append((new_exp, 2))
            elif e == 1:
                # Remove prime
                new_exp = current_exponents.copy()
                del new_exp[p]
                candidates.append((new_exp, 2))
                
        # 4. Replace p with q - try promising swaps first
        # Only try swaps that replace a larger prime with a smaller one (more likely beneficial)
        for p in used_primes:
            max_q = MAX_N * p // current_n
            # Try smaller primes first (more likely to fit and be beneficial)
            valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes and q < p]
            for q in valid_qs:
                new_exp = current_exponents.copy()
                del new_exp[p]
                new_exp[q] = 1
                candidates.append((new_exp, 3))
        
        # Sort by priority, then shuffle within same priority
        candidates.sort(key=lambda x: x[1])
        random.shuffle(candidates)
        
        # Evaluate candidates
        for new_exp, _ in candidates:
            if time.time() - start_time > DEADLINE:
                break
                
            s, n = evaluate(new_exp)
            if s > best_score:
                best_score = s
                best_n = n
                best_exponents = new_exp
                improved = True
                # Update current state for subsequent candidates in this batch
                current_exponents = new_exp
                current_n = n
                # Continue evaluating other candidates
        
    return best_n

construct()
