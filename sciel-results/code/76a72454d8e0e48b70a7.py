import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, keep 5s margin
    
    # 1. Generate primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    # 2. Generate candidate sets
    # We will try a few different sets of primes.
    # Set 1: Greedy smallest primes
    # Set 2: Greedy smallest primes, but try to fit one more by swapping?
    # Actually, let's just use the greedy set and a few random variations.
    
    def generate_greedy_set(primes):
        s = []
        sum_r = 0.0
        for p in primes:
            if sum_r + 1.0/p <= 1.0:
                s.append(p)
                sum_r += 1.0/p
        return s

    base_set = generate_greedy_set(primes)
    candidate_sets = [base_set]
    
    # Add some random sets from small primes
    small_primes = primes[:30]
    for _ in range(20):
        random.shuffle(small_primes)
        s = []
        sum_r = 0.0
        for p in small_primes:
            if sum_r + 1.0/p <= 1.0:
                s.append(p)
                sum_r += 1.0/p
        if s:
            candidate_sets.append(s)

    # 3. Optimization Function
    def get_L(prime_residues):
        # Returns the largest L such that 1..L are covered.
        # i.e., the first uncovered integer is L+1.
        # We check x=1, 2, ...
        x = 1
        while True:
            covered = False
            for p, r in prime_residues:
                if x % p == r:
                    covered = True
                    break
            if not covered:
                return x - 1
            x += 1
            if x > 100000: # Cap
                return 100000

    def optimize_residues(prime_set, max_time=10.0):
        # Initial residues: cover 1
        # x=1 is covered if r_p == 1 % p
        residues = [1 % p for p in prime_set]
        current_L = get_L(list(zip(prime_set, residues)))
        best_L_local = current_L
        best_residues_local = residues[:]
        
        T = 5.0
        cool = 0.995
        
        set_start = time.time()
        
        while time.time() - set_start < max_time:
            # Pick a prime to change
            idx = random.randint(0, len(prime_set) - 1)
            p = prime_set[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            residues[idx] = new_r
            new_L = get_L(list(zip(prime_set, residues)))
            
            if new_L > best_L_local:
                best_L_local = new_L
                best_residues_local = residues[:]
            
            delta = new_L - current_L
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or (T > 0 and random.random() < pow(2.71828, delta / T)):
                current_L = new_L
            else:
                residues[idx] = old_r
                
            T *= cool
            if T < 0.01:
                # Reheat if stuck?
                if time.time() - set_start < max_time * 0.8:
                    T = 2.0
                else:
                    break
        
        return best_L_local, best_residues_local

    best_L = 0
    best_set = []
    best_residues = []
    
    # Sort candidate sets by size (larger sets might be better) or just iterate
    for p_set in candidate_sets:
        if time.time() > deadline - 5:
            break
            
        # Allocate more time to the base set
        alloc_time = 20.0 if p_set == base_set else 5.0
        L, res = optimize_residues(p_set, max_time=alloc_time)
        
        if L > best_L:
            best_L = L
            best_set = p_set
            best_residues = res

    # 4. Final Verification
    sum_frac = Fraction(0, 1)
    for p in best_set:
        sum_frac += Fraction(1, p)
        
    if sum_frac > 1:
        # Fallback
        primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    result = []
    for p, r in zip(best_set, best_residues):
        result.append([p, r])
        
    return result
