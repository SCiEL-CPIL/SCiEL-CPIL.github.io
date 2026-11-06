import math
import time
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(200)

def _generate_divisors(primes, exponents):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d in current:
                divisors.append(d * pk)
    return divisors

def _tau_perp(primes, exponents):
    """Compute tau_perp score from prime factorization."""
    divisors = _generate_divisors(primes, exponents)
    divisors.sort()
    if len(divisors) < 2:
        return 0
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _compute_n(primes, exponents):
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    max_n = 10**18
    primes = _PRIMES
    
    # Find the largest primorial <= 10^18
    prod = 1
    max_k = 0
    for i, p in enumerate(primes):
        if prod * p <= max_n:
            prod *= p
            max_k = i + 1
        else:
            break
    
    # Base case: primorial of first max_k primes (all exponent 1)
    base_exps = [1] * max_k
    base_n = prod
    base_score = _tau_perp(primes[:max_k], base_exps)
    
    best_score = base_score
    best_n = base_n
    best_exps = base_exps[:]
    
    # Try variations:
    # 1. Remove one prime and add a different (larger) prime
    # 2. Square one prime (increase exponent from 1 to 2)
    # 3. Remove one prime and square another
    # 4. Remove two primes and add two different ones
    
    # Precompute primes beyond max_k for swapping
    available_primes = primes[max_k:max_k+50]  # primes just beyond the primorial
    
    # Strategy 1: Try squaring one prime in the primorial
    for i in range(max_k):
        if time.time() > deadline:
            break
        new_exps = base_exps[:]
        new_exps[i] = 2
        new_n = base_n * primes[i]
        if new_n <= max_n:
            score = _tau_perp(primes[:max_k], new_exps)
            if score > best_score:
                best_score = score
                best_n = new_n
                best_exps = new_exps[:]
    
    # Strategy 2: Remove prime i, add prime j (j > max_k)
    for i in range(max_k):
        if time.time() > deadline:
            break
        for j in available_primes:
            temp_n = base_n // primes[i] * j
            if temp_n > max_n:
                break
            # Build new exponent list
            new_exps = [0] * (max_k + 50)
            for k in range(max_k):
                if k != i:
                    new_exps[k] = 1
            new_exps[primes.index(j)] = 1
            # Only include primes that are actually used
            used_primes = [primes[k] for k in range(max_k) if k != i] + [j]
            used_exps = [1] * len(used_primes)
            # Need to map to the right prime indices
            # Simpler: just compute directly
            score = _tau_perp(used_primes, used_exps)
            if score > best_score:
                best_score = score
                best_n = temp_n
                best_exps = used_exps[:]
    
    # Strategy 3: Remove prime i, square prime j (j != i, both in first max_k)
    for i in range(max_k):
        if time.time() > deadline:
            break
        for j in range(max_k):
            if i == j:
                continue
            new_exps = base_exps[:]
            new_exps[i] = 0
            new_exps[j] = 2
            new_n = base_n // primes[i] * primes[j]
            if new_n <= max_n:
                score = _tau_perp(primes[:max_k], new_exps)
                if score > best_score:
                    best_score = score
                    best_n = new_n
                    best_exps = new_exps[:]
    
    # Strategy 4: Try primorial with one fewer prime but two primes squared
    if max_k >= 2:
        for i in range(max_k - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, max_k - 1):
                new_exps = base_exps[:max_k-1][:]
                new_exps[i] = 2
                new_exps[j] = 2
                new_n = 1
                for k in range(max_k - 1):
                    new_n *= primes[k] ** new_exps[k]
                if new_n <= max_n:
                    score = _tau_perp(primes[:max_k-1], new_exps)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        best_exps = new_exps[:]
    
    # Strategy 5: Local search from best found so far
    # Try single moves from best_exps
    if time.time() < deadline:
        current_exps = best_exps[:]
        current_n = best_n
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Determine which primes are used
            used_primes = [primes[i] for i, e in enumerate(current_exps) if e > 0]
            
            # Try increasing an exponent
            for i in range(len(current_exps)):
                if current_exps[i] >= 5:
                    continue
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = current_n * primes[i]
                if new_n <= max_n:
                    score = _tau_perp(primes[:len(current_exps)], new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
            
            # Try decreasing an exponent (if > 1)
            for i in range(len(current_exps)):
                if current_exps[i] <= 1:
                    continue
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = current_n // primes[i]
                if new_n <= max_n and new_n >= 2:
                    score = _tau_perp(primes[:len(current_exps)], new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
            
            # Try swapping: remove prime i (set to 0), add prime j (set to 1)
            for i in range(len(current_exps)):
                if current_exps[i] == 0:
                    continue
                for j in range(len(current_exps)):
                    if j == i or current_exps[j] > 0:
                        continue
                    new_exps = current_exps[:]
                    new_exps[i] = 0
                    new_exps[j] = 1
                    new_n = current_n // primes[i] * primes[j]
                    if 2 <= new_n <= max_n:
                        score = _tau_perp(primes[:len(current_exps)], new_exps)
                        if score > current_score:
                            current_exps = new_exps
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                                best_exps = current_exps[:]
    
    return best_n
