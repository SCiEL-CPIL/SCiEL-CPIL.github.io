import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate a list of small primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = []
    for i in range(2, 100):
        if is_prime(i):
            primes.append(i)
        if len(primes) >= 30:
            break
            
    def factorize(n):
        factors = []
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                factors.append((p, count))
        if n > 1:
            factors.append((n, 1))
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors:
            mult = 1
            new_divs = []
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * mult)
                mult *= p
            divs = new_divs
        return sorted(divs)

    def score(n):
        if n <= 1:
            return 0
        factors = factorize(n)
        divs = get_divisors(factors)
        if len(divs) < 2:
            return 0
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count

    # Initial best
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = score(best_n)
    
    # Local search function
    def local_search(n, max_iter=100):
        current_n = n
        current_score = score(current_n)
        factors = factorize(current_n)
        # Represent as list of (prime, exp)
        # We allow changes to primes and exponents
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Generate neighbors
            neighbors = []
            
            # 1. Change exponents
            for i, (p, e) in enumerate(factors):
                if e > 1:
                    # Decrement
                    new_factors = factors[:i] + [(p, e-1)] + factors[i+1:]
                    new_n = 1
                    for pp, ee in new_factors:
                        new_n *= pp**ee
                    if new_n <= 10**18:
                        neighbors.append((new_n, "dec_exp"))
                # Increment
                new_e = e + 1
                new_n = current_n * p
                if new_n <= 10**18:
                    neighbors.append((new_n, "inc_exp"))
            
            # 2. Replace a prime
            for i, (p, e) in enumerate(factors):
                # Try replacing p with next prime
                idx = primes.index(p)
                if idx + 1 < len(primes):
                    new_p = primes[idx+1]
                    # Check if new_p is already in factors
                    if all(pp != new_p for pp, _ in factors):
                        new_n = current_n // (p**e) * (new_p**e)
                        if new_n <= 10**18:
                            neighbors.append((new_n, "replace_prime"))
                # Try replacing p with previous prime
                if idx - 1 >= 0:
                    new_p = primes[idx-1]
                    if all(pp != new_p for pp, _ in factors):
                        new_n = current_n // (p**e) * (new_p**e)
                        if new_n <= 10**18:
                            neighbors.append((new_n, "replace_prime"))
            
            # 3. Add a prime
            # Find smallest prime not in factors
            existing_primes = set(p for p, _ in factors)
            for p in primes:
                if p not in existing_primes:
                    new_n = current_n * p
                    if new_n <= 10**18:
                        neighbors.append((new_n, "add_prime"))
                    break # Only try the smallest one to keep n small
            
            # 4. Remove a prime
            for i, (p, e) in enumerate(factors):
                new_factors = factors[:i] + factors[i+1:]
                if new_factors: # n must be > 1
                    new_n = 1
                    for pp, ee in new_factors:
                        new_n *= pp**ee
                    if new_n > 1:
                        neighbors.append((new_n, "remove_prime"))

            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_score = current_score
            
            # Shuffle to break ties
            random.shuffle(neighbors)
            
            for new_n, _ in neighbors:
                if new_n == current_n:
                    continue
                s = score(new_n)
                if s > best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor = new_n
            
            if best_neighbor is not None:
                current_n = best_neighbor
                current_score = best_neighbor_score
                factors = factorize(current_n)
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
        
        return current_n, current_score

    # Generate initial candidates
    candidates = []
    
    # Square free products
    for k in range(2, 16):
        n = 1
        for i in range(k):
            n *= primes[i]
            if n > 10**18:
                break
        if n > 1:
            candidates.append(n)
            
    # Random candidates
    for _ in range(50):
        n = 1
        num_primes = random.randint(2, 10)
        used = set()
        for _ in range(num_primes):
            p = random.choice(primes)
            if p in used:
                continue
            e = random.randint(1, 3)
            if n * (p**e) > 10**18:
                continue
            n *= p**e
            used.add(p)
        if n > 1:
            candidates.append(n)

    # Run local search on candidates
    for n in candidates:
        if time.time() > deadline:
            break
        local_search(n)

    return best_n
