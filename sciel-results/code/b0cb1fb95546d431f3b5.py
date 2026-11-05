import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18
    
    def calc_score(n):
        if n < 2:
            return 0
        # Factorize n
        temp = n
        factors = []
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
        
        # Generate divisors
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def estimate_coprime_score(exps):
        """
        Estimate the number of coprime adjacent pairs based on exponent structure.
        For n with k distinct prime factors each with exponent 1, the divisors form
        a hypercube. The number of coprime adjacencies is maximized when primes are
        small (more interleaving in sorted order).
        
        Key insight: for squarefree n with k primes, the score is approximately
        proportional to k * 2^(k-1) but limited by the actual sorted order.
        
        For non-squarefree, higher exponents reduce coprime adjacencies because
        more divisors share the same prime.
        """
        k = sum(1 for e in exps if e > 0)
        if k == 0:
            return 0
        
        # For squarefree: score is roughly the number of edges in the divisor lattice
        # where adjacent divisors in sorted order are coprime
        # Empirically, for primorial of first k primes, score grows superlinearly
        
        # Penalty for non-squarefree: each extra power of a prime reduces potential
        penalty = 0
        for e in exps:
            if e > 1:
                penalty += (e - 1) * 2
        
        # Base estimate: for k distinct primes, max coprime adjacencies ~ k * 2^(k-1)
        # but capped by total divisors - 1
        tau = 1
        for e in exps:
            tau *= (e + 1)
        
        # Theoretical upper bound on coprime adjacencies
        # Each divisor can be coprime to at most 2 neighbors
        # And two divisors are coprime iff they share no prime
        # For squarefree n with k primes: C(k,1) * 2^(k-1) is an upper bound
        upper_bound = min(tau - 1, k * (1 << (k - 1)) if k > 0 else 0)
        
        # Apply penalty for non-squarefree
        estimated = max(0, upper_bound - penalty * (1 << (k - 1)) if k > 0 else 0)
        
        return estimated
    
    # Search over the first 15 primes
    num_primes = 15
    
    # Start with primorial of first 15 primes
    start_exps = [1] * num_primes
    start_n = 1
    for i in range(num_primes):
        start_n *= primes[i]
    
    best_n = start_n
    best_score = calc_score(start_n)
    
    # Local search: try modifying exponents
    current_exps = list(start_exps)
    current_n = start_n
    current_score = best_score
    
    # Greedy local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_primes):
            # Try increasing exponent i
            new_exps = list(current_exps)
            new_exps[i] += 1
            new_n = current_n * primes[i]
            if new_n <= LIMIT:
                s = calc_score(new_n)
                if s > current_score:
                    current_exps = new_exps
                    current_n = new_n
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                    improved = True
            
            # Try decreasing exponent i
            if current_exps[i] > 0:
                new_exps = list(current_exps)
                new_exps[i] -= 1
                new_n = current_n // primes[i]
                if new_n >= 2:
                    s = calc_score(new_n)
                    if s > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_n = new_n
                        improved = True
        
        # Try adding a new prime (if we have room)
        for i in range(num_primes):
            if current_exps[i] == 0:
                new_exps = list(current_exps)
                new_exps[i] = 1
                new_n = current_n * primes[i]
                if new_n <= LIMIT:
                    s = calc_score(new_n)
                    if s > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_n = new_n
                        improved = True
    
    # Try different starting points: primorials of different sizes
    for k in range(10, 16):
        if time.time() >= deadline:
            break
        exps = [1] * k + [0] * (num_primes - k)
        n = 1
        for i in range(k):
            n *= primes[i]
        if n <= LIMIT:
            s = calc_score(n)
            if s > best_score:
                best_score = s
                best_n = n
    
    # Try some random perturbations
    import random
    random.seed(42)
    for _ in range(50):
        if time.time() >= deadline:
            break
        # Start from best and perturb
        exps = list(best_exps if 'best_exps' in dir() else current_exps)
        # Perturb 1-2 exponents
        for _ in range(random.randint(1, 2)):
            i = random.randint(0, num_primes - 1)
            exps[i] = max(0, min(5, exps[i] + random.randint(-1, 1)))
        n = 1
        for i in range(num_primes):
            if exps[i] > 0:
                n *= primes[i] ** exps[i]
        if 2 <= n <= LIMIT:
            s = calc_score(n)
            if s > best_score:
                best_score = s
                best_n = n
    
    return best_n
