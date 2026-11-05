import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Generate candidate integers composed of small primes
    # We focus on {2, 3} and {2, 3, 5} as the primary prime bases.
    # If we can find 12 distinct integers from {2,3} such that all pairwise sums
    # are also {2,3}-smooth (or {2,3,5}-smooth), we have a very low prime count.
    
    # Let's first try to construct a set where all pairwise sums are divisible only by 2 and 3.
    # Actually, a simpler heuristic: 
    # If we pick numbers of the form 2^a * 3^b, their sums might introduce new primes.
    # However, if we pick numbers that are all multiples of a large power of 2, 
    # say 2^k, then a+b = 2^k(x+y). The prime factors of a+b are {2} U factors(x+y).
    # If we can make x+y have very few prime factors, we win.
    
    # Strategy: 
    # 1. Try to find 12 numbers in [1, 10000] such that the set of prime factors of all pairwise sums is minimal.
    # 2. A known good structure for minimizing prime factors of sums is to use numbers that are "close" to each other or structured powers.
    # 3. Let's use a local search on a restricted pool of "smooth" numbers.
    
    # Generate pool of 2,3-smooth numbers up to 10000
    pool_23 = []
    for a in range(0, 15):
        for b in range(0, 15):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= 10000:
                pool_23.append(val)
    pool_23 = list(set(pool_23))
    
    # Also generate 2,3,5-smooth numbers
    pool_235 = []
    for a in range(0, 15):
        for b in range(0, 15):
            for c in range(0, 15):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= val <= 10000:
                    pool_235.append(val)
    pool_235 = list(set(pool_235))
    
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    def evaluate_set(numbers):
        if len(numbers) != 12:
            return float('inf')
        if len(set(numbers)) != 12:
            return float('inf')
        if max(numbers) > 10000 or min(numbers) < 1:
            return float('inf')
        
        prime_union = set()
        for i in range(12):
            for j in range(i + 1, 12):
                s = numbers[i] + numbers[j]
                prime_union.update(get_prime_factors(s))
        return len(prime_union)
    
    best_score = float('inf')
    best_set = None
    
    # Try to find a set within pool_23 that minimizes the prime union.
    # Since the pool is small-ish, we can try random sampling or greedy construction.
    # However, random sampling from a restricted pool might miss the structure.
    # Let's try a simple local search starting from random subsets of the pool.
    
    random.seed(42)
    
    # We will try to improve a random initial set from the pool_235 (larger pool for diversity)
    # but we expect the optimal solution to be very "smooth".
    
    # Heuristic: If we use numbers that are all powers of 2, e.g., 1, 2, 4, ..., 2048.
    # Sums: 2^a + 2^b = 2^min(a,b) * (1 + 2^|a-b|).
    # The term (1 + 2^k) might have prime factors.
    # 1+2=3, 1+4=5, 1+8=9(3), 1+16=17, 1+32=33(3,11)...
    # This introduces 5, 17, 11, etc.
    
    # What if we use numbers like 3^k?
    # 3^a + 3^b = 3^min(a,b) * (1 + 3^|a-b|).
    # 1+3=4(2), 1+9=10(2,5), 1+27=28(2,7), 1+81=82(2,41)...
    
    # What if we mix them?
    # Let's just run a stochastic local search on the pool_235.
    # We want to minimize the number of distinct primes in the sums.
    
    # Initial guess: Random selection from pool_235
    current_set = random.sample(pool_235, 12)
    current_score = evaluate_set(current_set)
    
    if current_score < best_score:
        best_score = current_score
        best_set = current_set[:]
    
    # Local Search
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        # Pick a random element to replace
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value from the pool, different from others
        new_val = random.choice(pool_235)
        if new_val in current_set and new_val != old_val:
            continue
            
        current_set[idx] = new_val
        new_score = evaluate_set(current_set)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Revert
            current_set[idx] = old_val
            
        # Early exit if we found a theoretically very good solution
        # The minimum possible is likely 2 (if all sums are 2,3-smooth) or 3.
        # If we hit 2, we can stop.
        if best_score == 2:
            break

    # If we didn't find a great solution, try a more targeted approach:
    # Force the set to be from pool_23 (stricter) and do a more exhaustive search if time permits.
    if best_score > 4 and time.time() < deadline - 10:
        # Greedy construction on pool_23
        # Start with 1, then add numbers that don't introduce new primes if possible
        # This is complex, so we stick to the local search result for now.
        pass

    if best_set is None:
        return list(range(1, 13))
        
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To adhere to the "keep function names" instruction, I should ensure construct is the entry point.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#     return list(range(1,13))
# So my construct() is the correct entry point.
# EVOLVE-BLOCK-END
