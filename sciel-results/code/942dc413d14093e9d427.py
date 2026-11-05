import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Sieve for primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    prime_set = set(primes)
    
    # We'll use a greedy approach
    # MAX_L is the maximum we'll try to cover
    MAX_L = 5000  # We'll try to cover up to this
    
    covered = [False] * (MAX_L + 2)
    classes = []
    used_primes = set()
    budget = Fraction(1, 1)
    current_sum = Fraction(0, 1)
    
    # For each n from 1 to MAX_L
    best_L = 0
    for n in range(1, MAX_L + 1):
        if time.time() > deadline:
            break
        
        if covered[n]:
            best_L = n
            continue
        
        # n is not covered, need to add a class
        # Try each unused prime, pick the one that gives best coverage
        best_p = None
        best_new_cov = -1
        best_r = None
        
        for p in primes:
            if p in used_primes:
                continue
            if current_sum + Fraction(1, p) > budget:
                continue
            
            r = n % p
            # Count new covers in [n, MAX_L]
            new_cov = 0
            for x in range(r if r >= n else r + ((n - r + p - 1) // p) * p, MAX_L + 1, p):
                if x >= n and not covered[x]:
                    new_cov += 1
            
            # Prefer primes that cover more new numbers
            # Tie-break: prefer larger p (cheaper)
            if new_cov > best_new_cov or (new_cov == best_new_cov and (best_p is None or p > best_p)):
                best_new_cov = new_cov
                best_p = p
                best_r = r
        
        if best_p is None:
            # Can't cover n, stop
            break
        
        # Add the class
        classes.append([best_p, best_r])
        used_primes.add(best_p)
        current_sum += Fraction(1, best_p)
        
        # Update covered
        r = best_r
        p = best_p
        for x in range(r if r >= 1 else r + p, MAX_L + 1, p):
            if 1 <= x <= MAX_L:
                covered[x] = True
        # Also check x = r if r >= 1
        if 1 <= r <= MAX_L:
            covered[r] = True
        
        best_L = n
    
    return classes
