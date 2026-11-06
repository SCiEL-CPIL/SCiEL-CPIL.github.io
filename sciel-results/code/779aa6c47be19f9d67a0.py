"""Optimized program: search for 12 distinct positive integers <=10000 minimizing distinct primes in pair sums."""
import time
import random
from itertools import combinations

# Precompute smallest-prime-factor table up to 20000 (max sum = 10000+9999=19999)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

def _prime_factors_bitmask(n):
    """Return bitmask of prime factors of n using SPF table."""
    mask = 0
    while n > 1:
        p = _spf[n]
        # Map prime p to a bit position. 
        # We can just use p itself if we treat it as an index, but bitsets need indices.
        # Since primes are sparse, we can map them to small indices.
        # However, for simplicity and speed in this context, let's stick to a set of primes 
        # but optimize the objective function to use a global set of "active" primes if possible?
        # No, bitmask is better. Let's map primes to indices 0..K-1.
        # We don't know K in advance, but it's small.
        # Let's just return a set of primes, but the objective function will be optimized.
        # Actually, the bottleneck is the set update. 
        # Let's try a different approach: 
        # Precompute for each sum the list of prime factors.
        # In objective, we can just iterate and add to a set. 
        # To make it faster, we can use a list of booleans or a set of integers.
        # Let's stick to sets but ensure the loop is tight.
        # Wait, the previous code used sets. Let's try to speed up the set update by 
        # using a local variable and updating at the end? No, union is needed.
        
        # Let's use a bitmask where bit i corresponds to the i-th prime number.
        # There are ~2262 primes below 20000. That's too big for a simple int bitmask on 32/64 bit, 
        # but Python ints are arbitrary precision. 
        # Let's map primes to indices.
        pass
    return mask

# Let's use a simpler optimization: 
# The number of distinct primes is small. 
# We can precompute the prime factors for each sum as a tuple of sorted primes.
# Then in objective, we can use a set.
# The main issue is the number of calls.

# Let's redefine the precomputation to store a set of primes, but we will optimize the search.
_prime_factors_cache = {}

def _get_prime_factors(n):
    if n in _prime_factors_cache:
        return _prime_factors_cache[n]
    factors = set()
    temp = n
    while temp > 1:
        p = _spf[temp]
        factors.add(p)
        while temp % p == 0:
            temp //= p
    _prime_factors_cache[n] = factors
    return factors

# Precompute for all sums
_pf_table = [None] * (_MAX_SUM + 1)
for _n in range(2, _MAX_SUM + 1):
    _pf_table[_n] = _get_prime_factors(_n)

def _objective(arr):
    """Count distinct primes dividing any sum of two distinct elements."""
    primes_set = set()
    n = len(arr)
    # Local variable for speed
    table = _pf_table
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            primes_set.update(table[ai + arr[j]])
    return len(primes_set)

# Track which primes are "essential" - appear in many sums
def _prime_frequency(arr):
    """Count how many pair sums each prime divides."""
    freq = {}
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            for p in _prime_factors_cached(s):
                freq[p] = freq.get(p, 0) + 1
    return freq

def _generate_structured_candidates():
    """Generate diverse structured candidate arrays with high probability of low prime union."""
    structured = []
    
    # 1. Arithmetic Progressions: a + i*d
    # Sums are 2a + (i+j)d. If d is small and a is chosen well, factors might be limited.
    for d in range(1, 200):
        for a in range(1, 200):
            arr = [a + i * d for i in range(12)]
            if max(arr) <= 10000 and len(set(arr)) == 12:
                structured.append(arr)
                
    # 2. Multiples of small bases
    # If all numbers are multiples of B, sums are multiples of B.
    # The prime factors of the sums will be Primes(B) U Primes(sums/B).
    for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 40, 48, 60]:
        max_mult = 10000 // base
        if max_mult < 12:
            continue
        # Regular multiples
        for start in range(1, 50):
            if start + 11 > max_mult:
                break
            arr = [base * (start + i) for i in range(12)]
            if len(set(arr)) == 12:
                structured.append(arr)
        
        # Random multiples
        for _ in range(20):
            mults = random.sample(range(1, max_mult + 1), 12)
            arr = [base * m for m in mults]
            if len(set(arr)) == 12:
                structured.append(arr)

    # 3. Numbers with specific modular properties
    # E.g., all numbers congruent to 1 mod 3 -> sums are 2 mod 3 (not divisible by 3)
    # All numbers odd -> sums even (divisible by 2)
    
    # Try to force sums to be divisible by only 2, 3, 5
    # If we pick numbers such that sums are multiples of 30, the prime factors of the quotients matter.
    
    # 4. Small integers
    # Often the optimal solution uses small numbers because large numbers introduce larger prime factors.
    for _ in range(150):
        arr = random.sample(range(1, 100), 12)
        structured.append(arr)
        
    for _ in range(150):
        arr = random.sample(range(2, 200, 2), 12) # All even
        structured.append(arr)

    for _ in range(150):
        arr = random.sample(range(3, 200, 3), 12) # All div by 3
        structured.append(arr)

    for _ in range(150):
        # Mix of small even and small multiples of 3
        candidates = list(range(2, 100, 2)) + list(range(3, 100, 3))
        candidates = list(set(candidates))
        arr = random.sample(candidates, 12)
        structured.append(arr)
    
    # 5. Numbers designed to have sums divisible by specific small primes
    # Try to construct sets where most pair sums are divisible by 2, 3, or 5
    for _ in range(100):
        # All numbers ≡ 1 mod 6 (sums ≡ 2 mod 6, so not divisible by 2 or 3)
        candidates = [x for x in range(1, 500) if x % 6 == 1]
        if len(candidates) >= 12:
            arr = random.sample(candidates, 12)
            structured.append(arr)
    
    for _ in range(100):
        # All numbers ≡ 2 mod 6 (sums ≡ 4 mod 6, so not divisible by 3)
        candidates = [x for x in range(2, 500) if x % 6 == 2]
        if len(candidates) >= 12:
            arr = random.sample(candidates, 12)
            structured.append(arr)
    
    for _ in range(100):
        # All numbers ≡ 0 mod 6 (sums divisible by 6)
        candidates = [x for x in range(6, 500, 6)]
        if len(candidates) >= 12:
            arr = random.sample(candidates, 12)
            structured.append(arr)
    
    for _ in range(100):
        # Mix: half ≡ 1 mod 6, half ≡ 5 mod 6 (sums ≡ 0 mod 6)
        candidates1 = [x for x in range(1, 500) if x % 6 == 1]
        candidates5 = [x for x in range(1, 500) if x % 6 == 5]
        if len(candidates1) >= 6 and len(candidates5) >= 6:
            arr = random.sample(candidates1, 6) + random.sample(candidates5, 6)
            if len(set(arr)) == 12:
                structured.append(arr)
    
    # 6. Numbers with specific modular properties for prime 5
    for _ in range(100):
        # All numbers ≡ 1 mod 5 (sums ≡ 2 mod 5, not divisible by 5)
        candidates = [x for x in range(1, 500) if x % 5 == 1]
        if len(candidates) >= 12:
            arr = random.sample(candidates, 12)
            structured.append(arr)
    
    for _ in range(100):
        # Half ≡ 1 mod 5, half ≡ 4 mod 5 (sums ≡ 0 mod 5)
        candidates1 = [x for x in range(1, 500) if x % 5 == 1]
        candidates4 = [x for x in range(1, 500) if x % 5 == 4]
        if len(candidates1) >= 6 and len(candidates4) >= 6:
            arr = random.sample(candidates1, 6) + random.sample(candidates4, 6)
            if len(set(arr)) == 12:
                structured.append(arr)

    return structured

def _local_search(current, current_score, deadline, best_score, best_arr):
    """Lightweight local search. Returns (best_score, best_arr)."""
    no_improve_count = 0
    while time.time() < deadline:
        no_improve_count += 1
        if no_improve_count > 20:
            break
        
        improved = False
        # Try a few random single-element replacements
        for _ in range(5):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            # Bias towards small numbers
            if random.random() < 0.8:
                new_val = random.randint(1, 200)
            else:
                new_val = random.randint(1, 10000)
            
            if new_val in current:
                continue
                
            cand = list(current)
            cand[i] = new_val
            if len(set(cand)) != 12 or max(cand) > 10000:
                continue
                
            score = _objective(cand)
            if score < current_score:
                current = cand
                current_score = score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(current)
                break
            elif score == current_score and random.random() < 0.1:
                current = cand
                improved = True
                break
        
        if not improved:
            # Try a swap
            if random.random() < 0.5:
                i, j = random.sample(range(12), 2)
                new_vi = random.randint(1, 200)
                new_vj = random.randint(1, 200)
                if new_vi != new_vj and new_vi not in current and new_vj not in current:
                    cand = list(current)
                    cand[i] = new_vi
                    cand[j] = new_vj
                    if len(set(cand)) == 12 and max(cand) <= 10000:
                        score = _objective(cand)
                        if score <= current_score:
                            current = cand
                            current_score = score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_arr = list(current)

        if improved:
            no_improve_count = 0
        else:
            # Perturb
            if random.random() < 0.2:
                i = random.randint(0, 11)
                cand = list(current)
                cand[i] = random.randint(1, 500)
                if len(set(cand)) == 12 and max(cand) <= 10000:
                    current = cand
                    current_score = _objective(current)

    return best_score, best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: must finish within 60s wall-clock
    
    # --- Phase 1: Structured Constructions ---
    best_arr = None
    best_score = float('inf')
    
    structured = _generate_structured_candidates()
    
    # Shuffle to avoid bias
    random.shuffle(structured)
    
    for arr in structured:
        if time.time() > deadline - 10:
            break
        arr = list(dict.fromkeys(arr))
        if len(arr) != 12:
            continue
        if any(x < 1 or x > 10000 for x in arr):
            continue
        score = _objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            if best_score <= 4:
                break
    
    if best_arr is None:
        # Fallback
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
    
    # --- Phase 2: Local Search from Best ---
    if time.time() < deadline - 5:
        current = list(best_arr)
        current_score = best_score
        local_deadline = min(deadline - 3, start_time + 20)
        best_score, best_arr = _local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 3: Random Restarts with Local Search ---
    if time.time() < deadline - 3:
        restarts_done = 0
        while time.time() < deadline - 2 and restarts_done < 50:
            restarts_done += 1
            
            # Generate diverse random candidates
            r = random.random()
            if r < 0.3:
                arr = random.sample(range(1, 100), 12)
            elif r < 0.6:
                arr = random.sample(range(2, 300, 2), 12)
            else:
                base = random.choice([2, 3, 6, 10])
                max_m = 10000 // base
                mults = random.sample(range(1, max_m + 1), 12)
                arr = [base * m for m in mults]
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            # Quick local search from this candidate
            if time.time() < deadline - 2:
                cur = list(arr)
                cur_score = score
                search_deadline = min(deadline - 2, time.time() + 0.5)
                best_score, best_arr = _local_search(cur, cur_score, search_deadline, best_score, best_arr)

    return best_arr

# EVOLVE-BLOCK-END
