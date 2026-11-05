import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # 1. Generate pool of 5-smooth numbers (primes 2, 3, 5) up to 10000
    pool = set()
    for a in range(0, 15):
        v2 = 2 ** a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3 ** b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                val = v23 * (5 ** c)
                if val > 10000:
                    break
                pool.add(val)
    pool = list(pool)
    random.shuffle(pool)
    
    # Precompute prime factors for all possible sums of pairs in the pool?
    # Pool size is ~300. Pairs ~45000.
    # Sums up to 20000.
    # We can just factor on the fly or precompute a sieve.
    
    # Sieve for smallest prime factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors
    
    # Initialize with a random subset of the pool
    current_set = random.sample(pool, 12)
    
    # Maintain prime counts
    # prime_count[p] = number of pairs (i,j) in current_set such that p divides a_i + a_j
    prime_counts = {}
    
    def update_prime_counts(add_val, remove_val):
        """
        Updates prime_counts when removing remove_val and adding add_val.
        Assumes current_set is updated BEFORE calling this? 
        No, better to pass the set excluding the removed element.
        """
        pass

    # Let's do a simpler approach: recompute cost if needed, or maintain counts carefully.
    # Given N=12, recomputing cost is 66 pairs. Factoring 66 numbers is trivial.
    # 66 * 10000 iterations = 660,000 factorizations.
    # With SPF, factorization is very fast.
    
    def evaluate_set(numbers):
        prime_union = set()
        for i in range(12):
            for j in range(i + 1, 12):
                s = numbers[i] + numbers[j]
                # Factor s
                n = s
                while n > 1:
                    p = spf[n]
                    prime_union.add(p)
                    while n % p == 0:
                        n //= p
        return len(prime_union)
    
    best_score = float('inf')
    best_set = None
    current_score = evaluate_set(current_set)
    
    if current_score < best_score:
        best_score = current_score
        best_set = current_set[:]
    
    # Simulated Annealing / Local Search
    # We want to minimize the number of distinct primes.
    
    temp = 1.0
    cooling = 0.999
    
    for step in range(200000):
        if time.time() > deadline:
            break
        
        # Pick random index to swap
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick new value from pool
        new_val = random.choice(pool)
        if new_val == old_val:
            continue
        if new_val in current_set:
            continue
            
        # Apply change
        current_set[idx] = new_val
        new_score = evaluate_set(current_set)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
            # Early exit if we hit the theoretical minimum for 5-smooth (3)
            # Or even 2 if we are lucky
            if best_score <= 2:
                break
        else:
            # Accept worse solution with probability p
            delta = new_score - current_score
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(-delta / temp)
            
            if random.random() < accept_prob:
                current_score = new_score
            else:
                # Revert
                current_set[idx] = old_val
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    # If we didn't find a good solution in the 5-smooth pool, 
    # maybe try a pool of 7-smooth numbers?
    # But 5-smooth is a good starting point.
    
    if best_set is None:
        return list(range(1, 13))
        
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# EVOLVE-BLOCK-END
