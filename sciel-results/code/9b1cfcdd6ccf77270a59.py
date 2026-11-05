import time
import math
from itertools import combinations, product

def _factorize(n):
    """Factorize n into (prime, exponent) pairs."""
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _score(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    factors = _factorize(n)
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    divisors.sort()
    
    # Count adjacent coprime pairs
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
    return score

def _score_squarefree(prime_set):
    """Fast score for squarefree numbers given a set of primes."""
    if not prime_set:
        return 0
    n = 1
    for p in prime_set:
        n *= p
    if n < 2 or n > 10**18:
        return 0
    return _score(n)

def construct():
    start = time.time()
    deadline = start + 145.0
    
    # Generate a good list of primes
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = []
    x = 2
    while len(primes) < 30:
        if is_prime(x):
            primes.append(x)
        x += 1 if x <= 2 else 2
    
    best_n = 2
    best_score = 0
    
    # Strategy 1: Find best squarefree number with many primes
    # For squarefree n with k primes, tau(n) = 2^k
    # The score is roughly proportional to 2^k but depends on arrangement
    
    # Try all subsets of size 13-15 from first 20 primes
    # But that's too many. Instead, try:
    # 1. Consecutive primes (smallest product for given count)
    # 2. Swap out largest prime for a slightly larger one to see if it helps
    
    # First, find the maximum k such that product of first k primes <= 10^18
    max_k = 0
    prod = 1
    for k in range(1, 20):
        prod *= primes[k-1]
        if prod <= 10**18:
            max_k = k
        else:
            break
    
    # Try all squarefree numbers with k = max_k, max_k-1, max_k-2 primes
    for k in range(max_k, max(max_k-3, 2), -1):
        if time.time() > deadline - 20:
            break
        
        # Try consecutive blocks of k primes
        for start_idx in range(0, len(primes) - k + 1):
            if time.time() > deadline - 20:
                break
            n = 1
            for i in range(k):
                n *= primes[start_idx + i]
            if n > 10**18:
                break
            s = _score(n)
            if s > best_score:
                best_score = s
                best_n = n
        
        # Try swapping the last prime with a larger one
        # Start with first k-1 primes, then try different last primes
        if k >= 3:
            base = 1
            for i in range(k-1):
                base *= primes[i]
            for last_idx in range(k, min(k+50, len(primes))):
                n = base * primes[last_idx]
                if n > 10**18:
                    break
                s = _score(n)
                if s > best_score:
                    best_score = s
                    best_n = n
    
    # Strategy 2: Try numbers with one prime having exponent 2
    # n = p^2 * q1 * q2 * ... * q_{k-1}
    # This gives tau = 3 * 2^{k-1}
    
    if time.time() < deadline - 15:
        for exp_prime_idx in range(0, min(8, len(primes))):
            if time.time() > deadline - 15:
                break
            p = primes[exp_prime_idx]
            p2 = p * p
            
            # Find max k such that p^2 * (product of k other primes) <= 10^18
            max_k2 = 0
            prod = p2
            for k in range(1, 18):
                # Use the smallest primes not equal to p
                other_primes = [pr for i, pr in enumerate(primes) if i != exp_prime_idx]
                if k > len(other_primes):
                    break
                new_prod = prod
                for j in range(k):
                    new_prod *= other_primes[j]
                    if new_prod > 10**18:
                        break
                if new_prod <= 10**18:
                    max_k2 = k
                else:
                    break
            
            # Try various combinations
            for k in range(max(max_k2-2, 1), max_k2+1):
                if time.time() > deadline - 15:
                    break
                other_primes = [pr for i, pr in enumerate(primes) if i != exp_prime_idx]
                # Try consecutive blocks from other_primes
                for start_idx in range(0, min(len(other_primes) - k + 1, 10)):
                    if time.time() > deadline - 15:
                        break
                    n = p2
                    for i in range(k):
                        n *= other_primes[start_idx + i]
                    if n > 10**18:
                        break
                    s = _score(n)
                    if s > best_score:
                        best_score = s
                        best_n = n
    
    return best_n
