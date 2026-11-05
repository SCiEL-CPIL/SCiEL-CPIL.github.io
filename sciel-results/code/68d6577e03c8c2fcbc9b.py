import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute primes up to a reasonable limit
    # We need primes to form n <= 10^18.
    # Product of first 15 primes is ~6.1e17.
    # Product of first 16 primes is ~2.2e19 (too big).
    # So we are likely dealing with 14-15 distinct primes.
    # We might swap small primes for larger ones if it improves score.
    # Let's sieve up to 5000 to have plenty of candidates for swapping.
    
    def get_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = get_primes(5000)
    prime_set = set(primes)

    # Fast scoring function
    # For a given n, factorize, generate divisors, sort, count coprime adjacent pairs.
    # Optimization: If n is squarefree, divisors are subsets of primes.
    # We can represent divisors as bitmasks if k <= 20.
    # But k is around 15. 2^15 = 32768 divisors.
    # Generating and sorting 32k integers is fast in Python (~5-10ms).
    # The bottleneck in the current program is likely the factorization and the loop overhead.
    # We can cache scores for specific prime sets? No, the set changes.
    # We can speed up factorization by using the known prime list.
    
    def calc_score_fast(prime_factors):
        # prime_factors: list of (p, e)
        # Generate divisors
        divs = [1]
        for p, e in prime_factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        divs.sort()
        
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    def get_factors(n):
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        return factors

    def score_n(n):
        factors = get_factors(n)
        return calc_score_fast(factors)

    # Initialize with a good guess: Primorial of first 15 primes
    current_primes = primes[:15]
    n_current = 1
    for p in current_primes:
        n_current *= p
        
    # Ensure n_current <= 10^18
    while n_current > 10**18 and current_primes:
        current_primes.pop()
        n_current = 1
        for p in current_primes:
            n_current *= p

    best_n = n_current
    best_score = score_n(best_n)
    current_n = best_n
    current_score = best_score

    # Simulated Annealing
    # State: set of primes (with exponent 1, since squarefree is likely optimal for this metric)
    # We will represent state as a set of prime indices or just the primes themselves.
    # Moves:
    # 1. Swap a prime in the set with a prime not in the set.
    # 2. Add a prime (if product allows).
    # 3. Remove a prime.
    
    # To make SA efficient, we need to evaluate scores quickly.
    # We will maintain the current set of primes.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995 # Cooling rate
    
    # We will run for a fixed number of steps or until time runs out.
    # Let's estimate steps.
    # Each step involves generating divisors (~32k items), sorting, and GCDs.
    # This takes ~5-10ms in Python.
    # 140s / 0.01s = 14000 steps.
    # This is a reasonable number for SA.
    
    # To speed up, we can precompute divisors for the current set?
    # No, the set changes.
    
    # Optimization: 
    # Instead of full factorization and divisor generation every time,
    # we can maintain the list of divisors?
    # Swapping a prime p for q:
    # Divisors are subsets of S. 
    # If we replace p with q, the values of divisors containing p change.
    # This is complex to update incrementally without re-sorting.
    # Re-sorting 32k items is O(N log N) ~ 32k * 15 ~ 500k ops.
    # Python is slow. 500k ops might take 0.1s? No, list.sort is C-optimized.
    # Let's assume 5ms per evaluation.
    
    # Strategy: 
    # 1. Start with primorial.
    # 2. Run SA.
    
    # To improve speed, we can restrict the search space.
    # It is known that for maximizing coprime adjacent divisors, 
    # the primes should be "small" to keep the number of divisors high (2^k),
    # but "spread out" to ensure coprime-ness?
    # Actually, gcd(d_i, d_{i+1}) == 1 is easier if they share no primes.
    # In a sorted list of squarefree divisors, adjacent elements often share primes.
    # Example: 1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35...
    # 1-2: gcd 1. 2-3: gcd 1. 3-5: gcd 1. 5-6: gcd 1. 6-7: gcd 1. 7-10: gcd 1.
    # 10-14: gcd 2. 14-15: gcd 1. 15-21: gcd 3. 21-30: gcd 3. 30-35: gcd 5.
    # Score for 2*3*5*7 = 30: Divs: 1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210.
    # Pairs:
    # 1-2 (1), 2-3 (1), 3-5 (1), 5-6 (1), 6-7 (1), 7-10 (1), 10-14 (2, no), 14-15 (1), 15-21 (3, no), 21-30 (3, no), 30-35 (5, no), 35-42 (7, no), 42-70 (14, no), 70-105 (35, no), 105-210 (105, no).
    # Score: 7.
    
    # The score is roughly proportional to the number of divisors, but penalized by clustering.
    
    # Let's implement the SA loop.
    
    steps = 0
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a neighbor
        # Move type: Swap (most likely), Add, Remove
        move_type = random.random()
        new_primes = current_primes[:]
        new_n = current_n
        
        if move_type < 0.7: # Swap
            if len(current_primes) < 2:
                continue
            i = random.randint(0, len(current_primes) - 1)
            p_old = current_primes[i]
            
            # Find a candidate to swap in
            # Max new prime q such that new_n <= 10^18
            # new_n = current_n // p_old * q <= 10^18
            # q <= 10^18 * p_old // current_n
            max_q = int((10**18 * p_old) // current_n)
            
            # Choose a random prime from primes not in current_primes and <= max_q
            # To avoid bias, we can pick a random index from the list of valid primes
            valid_candidates = [p for p in primes if p not in current_primes and p <= max_q]
            if not valid_candidates:
                continue
            p_new = random.choice(valid_candidates)
            
            new_primes[i] = p_new
            new_n = current_n // p_old * p_new
            
        elif move_type < 0.85: # Add
            # Find a prime q not in current_primes such that current_n * q <= 10^18
            max_q = int(10**18 // current_n)
            valid_candidates = [p for p in primes if p not in current_primes and p <= max_q]
            if not valid_candidates:
                continue
            p_new = random.choice(valid_candidates)
            new_primes.append(p_new)
            new_n = current_n * p_new
            
        else: # Remove
            if len(current_primes) < 2:
                continue
            i = random.randint(0, len(current_primes) - 1)
            p_old = current_primes[i]
            new_primes.pop(i)
            new_n = current_n // p_old
            
        # Evaluate
        new_score = score_n(new_n)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_n = new_n
            current_primes = new_primes
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
        
        T *= alpha
        steps += 1

    return best_n
