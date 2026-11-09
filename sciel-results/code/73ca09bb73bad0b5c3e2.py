import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    
    # Generate primes up to ~200
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = []
    p = 2
    while p <= 300:
        if is_prime(p):
            primes.append(p)
        p += 1
    
    # Find k: largest such that product of first k primes <= limit
    n = 1
    k = 0
    for p in primes:
        if n * p > limit:
            break
        n *= p
        k += 1
    
    initial_set = set(primes[:k])
    
    def score_prime_set(prime_set):
        """Compute tau_perp for squarefree number with given prime set."""
        if not prime_set:
            return 0
        plist = sorted(prime_set)
        divisors = [1]
        for p in plist:
            divisors = divisors + [d * p for d in divisors]
        divisors.sort()
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count
    
    best_score = score_prime_set(initial_set)
    best_set = initial_set.copy()
    
    # Single swap search
    for i in range(k):
        if time.time() > deadline - 30:
            break
        for j in range(k, min(len(primes), k + 15)):
            if time.time() > deadline - 30:
                break
            new_set = set(initial_set)
            new_set.discard(primes[i])
            new_set.add(primes[j])
            s = score_prime_set(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set.copy()
    
    # Double swap search with limited range
    max_j = min(len(primes), k + 10)
    for i in range(k):
        if time.time() > deadline - 15:
            break
        for j in range(i+1, k):
            if time.time() > deadline - 15:
                break
            for a in range(k, max_j):
                if time.time() > deadline - 15:
                    break
                for b in range(a+1, max_j):
                    if time.time() > deadline - 15:
                        break
                    new_set = set(initial_set)
                    new_set.discard(primes[i])
                    new_set.discard(primes[j])
                    new_set.add(primes[a])
                    new_set.add(primes[b])
                    s = score_prime_set(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set.copy()
    
    # Compute final n
    n = 1
    for p in sorted(best_set):
        n *= p
    
    return n
