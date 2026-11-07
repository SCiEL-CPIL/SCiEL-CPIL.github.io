"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for each number 1..256
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        p = 2
        while p * p <= x:
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
            p += 1
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    # Precompute compatibility: can[i][j] = True if gcd(i,j)==1
    # We'll use prime factor intersection for speed
    # Actually, let's precompute a bitmask of small primes for each number
    # Small primes are those <= sqrt(256) = 16: [2,3,5,7,11,13]
    # For large primes > 16, each number has at most one such prime factor
    # since 17*17 > 256.
    
    small_primes = [2, 3, 5, 7, 11, 13]
    small_prime_index = {p: i for i, p in enumerate(small_primes)}
    
    # For each number, store:
    # - bitmask of small prime factors (6 bits)
    # - the large prime factor (>16) if any, else 0
    small_mask = [0] * (N + 1)
    large_prime_factor = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        lp = 0
        for p in prime_factors[i]:
            if p in small_prime_index:
                mask |= (1 << small_prime_index[p])
            else:
                lp = p
        small_mask[i] = mask
        large_prime_factor[i] = lp
    
    # Two numbers i, j are coprime iff:
    # 1. small_mask[i] & small_mask[j] == 0
    # 2. large_prime_factor[i] != large_prime_factor[j] (unless both are 0)
    # 3. If one has large prime factor L and the other has small prime factor L... 
    #    Wait, L > 16, and small primes are <= 13, so no overlap.
    #    But what if number i has large prime factor 17, and number j is 17?
    #    Then j also has large_prime_factor = 17. So condition 2 covers it.
    #    What if i = 34 (2*17) and j = 17? 
    #    i: small_mask has bit for 2, large_prime_factor = 17
    #    j: small_mask = 0, large_prime_factor = 17
    #    small_mask[i] & small_mask[j] = 0, but large_prime_factor[i] == large_prime_factor[j] = 17
    #    So they are NOT coprime. Correct.
    
    def are_coprime(i, j):
        if i == j:
            return False
        if small_mask[i] & small_mask[j]:
            return False
        if large_prime_factor[i] != 0 and large_prime_factor[i] == large_prime_factor[j]:
            return False
        return True
    
    # Strategy: 
    # 1. Start with all primes (they're all pairwise coprime)
    # 2. Try to replace small primes with larger composites that are coprime to all others
    # 3. Use local search / simulated annealing to improve
    
    # Get all primes up to 256
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, N + 1) if is_prime(x)]
    
    # Initial solution: 1 + all primes
    current_set = set([1] + primes)
    current_sum = 1 + sum(primes)
    best_set = set(current_set)
    best_sum = current_sum
    
    # Precompute for each number, which other numbers it conflicts with
    # conflict_mask[i] = set of j != i such that not coprime(i, j)
    # This is expensive to precompute fully, so we'll check on the fly
    
    # Local search: try to swap a number in the set with a number not in the set
    # or add a number if possible
    
    def can_add(num, s):
        """Check if num can be added to set s"""
        if num in s:
            return False
        for x in s:
            if not are_coprime(num, x):
                return False
        return True
    
    def try_add(num, s, s_sum):
        """Try to add num to s, return (success, new_set, new_sum)"""
        if num in s or num > N:
            return False, s, s_sum
        for x in s:
            if not are_coprime(num, x):
                return False, s, s_sum
        return True, s | {num}, s_sum + num
    
    def try_swap(old, new, s, s_sum):
        """Try to swap old for new in s"""
        if new in s or new > N or new == old:
            return False, s, s_sum
        s_without_old = s - {old}
        for x in s_without_old:
            if not are_coprime(new, x):
                return False, s, s_sum
        return True, s_without_old | {new}, s_sum - old + new
    
    # Phase 1: Greedy addition of composites
    # Sort all numbers by value descending
    all_numbers = list(range(N, 0, -1))
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        progress = False
        for num in all_numbers:
            if time.time() > deadline:
                break
            if num in current_set:
                continue
            if can_add(num, current_set):
                current_set.add(num)
                current_sum += num
                progress = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = set(current_set)
        if not progress:
            break
    
    # Phase 2: Local search with swaps
    random.seed(42)
    temperature = 100.0
    cooling = 0.999
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random element to swap out
        elements = list(current_set)
        if not elements:
            break
        old = random.choice(elements)
        
        # Pick a random candidate to swap in
        candidates = [x for x in range(2, N + 1) if x not in current_set]
        if not candidates:
            break
        new = random.choice(candidates)
        
        success, new_set, new_sum = try_swap(old, new, current_set, current_sum)
        
        if success:
            delta = new_sum - current_sum
            if delta >= 0 or random.random() < math.exp(delta / temperature):
                current_set = new_set
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = set(current_set)
        
        temperature *= cooling
    
    # Phase 3: More aggressive local search - try all possible swaps
    if time.time() < deadline - 10:
        for iteration in range(100):
            if time.time() > deadline:
                break
            progress = False
            elements = list(best_set)
            for old in elements:
                if time.time() > deadline:
                    break
                for new in range(2, N + 1):
                    if new in best_set or new == old:
                        continue
                    success, new_set, new_sum = try_swap(old, new, best_set, best_sum)
                    if success and new_sum > best_sum:
                        best_set = new_set
                        best_sum = new_sum
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break
    
    # Phase 4: Try adding 1 if not present
    if 1 not in best_set:
        if can_add(1, best_set):
            best_set.add(1)
            best_sum += 1
    
    # Final verification
    result = list(best_set)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if not are_coprime(result[i], result[j]):
                # Fallback to primes
                result = [1] + primes
                best_sum = 1 + sum(primes)
                break
        else:
            continue
        break
    
    # Ensure distinct, in range, correct types
    result = list(set(result))
    result = [int(x) for x in result if 1 <= x <= N]
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
