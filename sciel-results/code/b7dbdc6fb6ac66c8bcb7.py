"""Search for 12 distinct positive integers <=10000 minimizing distinct prime factors of pairwise sums."""
import time
import random
from itertools import combinations

def _build_spf(limit):
    """Return smallest prime factor array up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _factor_set(n, spf):
    """Return set of prime factors of n using SPF table."""
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score(arr, spf):
    """Return the set of all distinct prime factors across all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            m = s
            while m > 1:
                p = spf[m]
                primes.add(p)
                while m % p == 0:
                    m //= p
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    max_sum = 20000
    spf = _build_spf(max_sum)

    best_score = float('inf')
    best_arr = None

    def update(arr):
        nonlocal best_score, best_arr
        primes = _score(arr, spf)
        if len(primes) < best_score:
            best_score = len(primes)
            best_arr = arr
        return len(primes)

    # Precompute prime factors for all numbers up to 20000 for speed
    # Actually, let's just use the SPF table directly in scoring.
    
    # Strategy 1: Structured sets - all numbers divisible by a common factor
    # If all numbers are multiples of d, all sums are multiples of d.
    # We want the remaining part (sum/d) to have few prime factors.
    
    # Try multiples of various bases
    bases = [6, 12, 24, 30, 60, 120, 2, 4, 8, 16, 3, 9, 18, 54]
    
    for base in bases:
        if time.time() > deadline - 10:
            break
        mults = [base * k for k in range(1, 10000 // base + 1) if base * k <= 10000]
        if len(mults) < 12:
            continue
        
        # Try random subsets
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            arr = random.sample(mults, 12)
            s = update(arr)
            if s <= 2:
                return arr
    
    # Strategy 2: Powers of 2 and 3 (smooth numbers)
    smooths = set()
    for a in range(14):
        for b in range(9):
            val = (1 << a) * (3 ** b)
            if 1 <= val <= 10000:
                smooths.add(val)
    smooths = sorted(smooths)
    
    if len(smooths) >= 12:
        for trial in range(500):
            if time.time() > deadline - 10:
                break
            arr = random.sample(smooths, 12)
            s = update(arr)
            if s <= 2:
                return arr
    
    # Strategy 3: Local search / hill climbing
    # Start from best found so far or a reasonable initial guess
    if best_arr is None:
        best_arr = [6 * k for k in range(1, 13)]
    
    current = list(best_arr)
    current_score = best_score
    
    # Build a pool of candidate replacements
    all_numbers = set(range(1, 10001))
    
    # Hill climbing: try to replace one element at a time
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        if current_score <= 2:
            break
        
        improved = False
        current_set = set(current)
        
        # Try each position
        for idx in range(12):
            if time.time() > deadline - 5:
                break
            
            old_val = current[idx]
            # Try a few random replacements
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                new_val = random.randint(1, 10000)
                if new_val in current_set and new_val != old_val:
                    continue
                
                new_arr = current[:]
                new_arr[idx] = new_val
                
                # Quick check: does this improve?
                # Only recompute affected sums
                new_primes = set()
                for i in range(12):
                    for j in range(i + 1, 12):
                        s = new_arr[i] + new_arr[j]
                        m = s
                        while m > 1:
                            p = spf[m]
                            new_primes.add(p)
                            while m % p == 0:
                                m //= p
                
                if len(new_primes) < current_score:
                    current = new_arr
                    current_score = len(new_primes)
                    current_set = set(current)
                    improved = True
                    if current_score <= 2:
                        return current
                    break
        
        if not improved:
            # Perturbation: randomly change 1-2 elements
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, 11)
                current[idx] = random.randint(1, 10000)
            # Ensure distinctness
            while len(set(current)) < 12:
                dup = random.choice(current)
                current[current.index(dup)] = random.randint(1, 10000)
            current_score = _score(current, spf)
    
    # Final validation
    if best_arr is None or len(best_arr) != 12 or len(set(best_arr)) != 12 or any(x < 1 or x > 10000 for x in best_arr):
        best_arr = list(range(1, 13))
    
    # Update best if local search improved
    if current_score < best_score:
        best_score = current_score
        best_arr = current
    
    return best_arr
