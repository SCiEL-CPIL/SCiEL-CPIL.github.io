import time
import random
import math
from itertools import combinations

# Precompute Smallest Prime Factor (SPF) up to 20000
MAX_VAL = 10000
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:  # i is prime
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    """Return set of prime factors of n using SPF."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factors as bitmasks for all numbers up to MAX_SUM
# We'll use the first 64 primes as our "universe" for bitmask representation
primes_list = []
for i in range(2, MAX_SUM + 1):
    if spf[i] == i:
        primes_list.append(i)
        if len(primes_list) >= 64:
            break

prime_to_bit = {p: (1 << i) for i, p in enumerate(primes_list)}

# Precompute bitmask of prime factors for each number
pf_mask = [0] * (MAX_SUM + 1)
for i in range(2, MAX_SUM + 1):
    mask = 0
    n = i
    while n > 1:
        p = spf[n]
        if p in prime_to_bit:
            mask |= prime_to_bit[p]
        while n % p == 0:
            n //= p
    pf_mask[i] = mask

def count_bits(x):
    """Count set bits in an integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_score = float('inf')
    
    def calc_score_mask(nums):
        """Calculate score using bitmask OR operation."""
        mask = 0
        n = len(nums)
        for i in range(n):
            xi = nums[i]
            for j in range(i+1, n):
                s = xi + nums[j]
                mask |= pf_mask[s]
        return count_bits(mask)
    
    def is_smooth_wrt(s, target_mask):
        """Check if s's prime factors are all in target_mask."""
        return (pf_mask[s] & ~target_mask) == 0
    
    # Strategy 1: Target prime set approach
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for k in range(2, min(len(small_primes), 6)):
        if time.time() > deadline:
            break
        
        for prime_combo in combinations(small_primes, k):
            if time.time() > deadline:
                break
            
            target_mask = 0
            for p in prime_combo:
                if p in prime_to_bit:
                    target_mask |= prime_to_bit[p]
            
            M = 1
            for p in prime_combo:
                M *= p
            
            if M > MAX_VAL // 2:
                continue
            
            # Try arithmetic progression: a, a+M, ..., a+11M
            for a in range(1, min(M + 1, MAX_VAL - 11*M + 1)):
                if time.time() > deadline:
                    break
                candidate = [a + i * M for i in range(12)]
                if candidate[-1] > MAX_VAL:
                    continue
                valid = True
                for i in range(12):
                    for j in range(i+1, 12):
                        if not is_smooth_wrt(candidate[i] + candidate[j], target_mask):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    sc = calc_score_mask(candidate)
                    if sc < best_score:
                        best_score = sc
                        best_set = candidate[:]
                    break
            
            # Try multiples: M, 2M, ..., 12M
            if 12 * M <= MAX_VAL:
                candidate = [i * M for i in range(1, 13)]
                valid = True
                for i in range(12):
                    for j in range(i+1, 12):
                        if not is_smooth_wrt(candidate[i] + candidate[j], target_mask):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    sc = calc_score_mask(candidate)
                    if sc < best_score:
                        best_score = sc
                        best_set = candidate[:]
            
            # Try: a, 2a, ..., 12a where a is P-smooth
            for a in range(1, min(M + 1, MAX_VAL // 12 + 1)):
                if time.time() > deadline:
                    break
                if not is_smooth_wrt(a, target_mask):
                    continue
                candidate = [i * a for i in range(1, 13)]
                if candidate[-1] > MAX_VAL:
                    continue
                valid = True
                for i in range(12):
                    for j in range(i+1, 12):
                        if not is_smooth_wrt(candidate[i] + candidate[j], target_mask):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    sc = calc_score_mask(candidate)
                    if sc < best_score:
                        best_score = sc
                        best_set = candidate[:]
            
            if best_score <= k:
                break
        
        if best_score <= k:
            break
    
    # Strategy 2: Local search with smooth number bias
    if best_score > 4:
        smooth_bases = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 48, 60, 72, 90, 120, 144, 180, 240, 360, 720]
        
        current = []
        for _ in range(12):
            if random.random() < 0.6:
                base = random.choice(smooth_bases)
                mult = random.randint(1, MAX_VAL // base)
                val = base * mult
            else:
                val = random.randint(1, MAX_VAL)
            while val in current:
                if random.random() < 0.6:
                    base = random.choice(smooth_bases)
                    mult = random.randint(1, MAX_VAL // base)
                    val = base * mult
                else:
                    val = random.randint(1, MAX_VAL)
            current.append(val)
        
        current_score = calc_score_mask(current)
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
        
        no_improve = 0
        max_no_improve = 300
        
        while no_improve < max_no_improve and time.time() < deadline:
            i = random.randint(0, 11)
            old_val = current[i]
            
            if random.random() < 0.5:
                base = random.choice(smooth_bases)
                mult = random.randint(1, MAX_VAL // base)
                new_val = base * mult
            else:
                new_val = random.randint(1, MAX_VAL)
                
            if new_val == old_val or new_val in current:
                no_improve += 1
                continue
                
            current[i] = new_val
            new_score = calc_score_mask(current)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
                    if best_score <= 3:
                        break
            else:
                current[i] = old_val
                no_improve += 1
    
    # Strategy 3: More restarts if needed
    if best_score > 4:
        for r in range(30):
            if time.time() > deadline:
                break
                
            current = []
            for _ in range(12):
                if random.random() < 0.6:
                    base = random.choice(smooth_bases)
                    mult = random.randint(1, MAX_VAL // base)
                    val = base * mult
                else:
                    val = random.randint(1, MAX_VAL)
                while val in current:
                    if random.random() < 0.6:
                        base = random.choice(smooth_bases)
                        mult = random.randint(1, MAX_VAL // base)
                        val = base * mult
                    else:
                        val = random.randint(1, MAX_VAL)
                current.append(val)
            
            current_score = calc_score_mask(current)
            
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                if best_score <= 3:
                    break
            
            no_improve = 0
            max_no_improve = 150
            
            while no_improve < max_no_improve and time.time() < deadline:
                i = random.randint(0, 11)
                old_val = current[i]
                
                if random.random() < 0.5:
                    base = random.choice(smooth_bases)
                    mult = random.randint(1, MAX_VAL // base)
                    new_val = base * mult
                else:
                    new_val = random.randint(1, MAX_VAL)
                    
                if new_val == old_val or new_val in current:
                    no_improve += 1
                    continue
                    
                current[i] = new_val
                new_score = calc_score_mask(current)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
                        if best_score <= 3:
                            break
                else:
                    current[i] = old_val
                    no_improve += 1
    
    return best_set

# EVOLVE-BLOCK-END
