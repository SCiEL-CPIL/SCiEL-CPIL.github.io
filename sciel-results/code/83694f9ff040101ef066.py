import math
import time
import random

MAX_N = 10**18
DEADLINE = 140.0

# Generate primes up to 1000 to allow swapping in larger primes if needed
def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = generate_primes(1000)
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
    # We maintain a set of candidate moves.
    # Moves: 
    # 1. Increase exponent of p by 1
    # 2. Decrease exponent of p by 1 (if > 1)
    # 3. Replace p with q (q not in exponents, or e_q=0)
    # 4. Add new prime q
    # 5. Remove prime p
    
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
                
        # Generate candidate moves
        candidates = []
        
        # 1. Increase exponent
        for p, e in current_exponents.items():
            if e > 0:
                if current_n * p <= MAX_N:
                    new_exp = current_exponents.copy()
                    new_exp[p] = e + 1
                    candidates.append(new_exp)
                    
        # 2. Decrease exponent
        for p, e in current_exponents.items():
            if e > 1:
                new_exp = current_exponents.copy()
                new_exp[p] = e - 1
                candidates.append(new_exp)
            elif e == 1:
                # Remove prime
                new_exp = current_exponents.copy()
                del new_exp[p]
                candidates.append(new_exp)
                
        # 3. Replace p with q
        # To limit search space, only consider swapping with primes close in value or small primes
        # If p is small, we might want to swap with a larger prime to free up "space"?
        # Actually, if we swap small p for large q, n increases.
        # If we swap large p for small q, n decreases.
        
        used_primes = set(p for p, e in current_exponents.items() if e > 0)
        
        # Try swapping each used prime with a few unused primes
        # We can't try all 168 primes for each of 15 used primes (2520 swaps) in every step if it's slow.
        # But evaluation is ~10ms. 2520 * 10ms = 25s per iteration. Too slow.
        # We need to prune candidates.
        
        # Strategy: Only try swaps that keep n within MAX_N.
        # And maybe only try a random subset of swaps to keep it fast?
        # Or, sort primes and only try nearby swaps?
        
        # Let's try a limited set of swaps:
        # For each p in used_primes:
        #   For each q in PRIMES (not in used_primes):
        #     if current_n * q // p <= MAX_N:
        #        add candidate
        
        # This is O(K * M). K=15, M=168. ~2500 candidates.
        # If we evaluate all, it's too slow.
        # We will randomly sample 100 swaps per iteration.
        
        swap_candidates = []
        for p in used_primes:
            # Find potential q
            # q must be such that n * q / p <= MAX_N => q <= MAX_N * p / n
            max_q = MAX_N * p // current_n
            # Filter PRIMES for q <= max_q and q not in used_primes
            # And q != p
            valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
            if valid_qs:
                # Randomly pick a few
                for q in random.sample(valid_qs, min(5, len(valid_qs))):
                    new_exp = current_exponents.copy()
                    new_exp[p] = 0 # Remove p
                    if q in new_exp:
                        new_exp[q] += 1
                    else:
                        new_exp[q] = 1
                    swap_candidates.append(new_exp)
        
        # 4. Add new prime
        # q such that n * q <= MAX_N
        max_q = MAX_N // current_n
        valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
        if valid_qs:
            for q in random.sample(valid_qs, min(5, len(valid_qs))):
                new_exp = current_exponents.copy()
                new_exp[q] = 1
                candidates.append(new_exp)

        candidates.extend(swap_candidates)
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        # Evaluate candidates
        for new_exp in candidates:
            if time.time() - start_time > DEADLINE:
                break
                
            s, n = evaluate(new_exp)
            if s > best_score:
                best_score = s
                best_n = n
                best_exponents = new_exp
                improved = True
                # Restart search from new best
                # Break to restart while loop
                break
                
    return best_n

construct()
