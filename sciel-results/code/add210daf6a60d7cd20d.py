import time
import random
import math

MAX_VAL = 10000
MAX_SUM = 20000

# Precompute SPF
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    best_set = None
    best_k = float('inf')
    
    # Fallback best
    fallback_best_score = float('inf')
    fallback_best_set = None
    
    def calc_raw_score(nums):
        primes_found = set()
        for i in range(12):
            for j in range(i+1, 12):
                s = nums[i] + nums[j]
                primes_found |= get_prime_factors(s)
        return len(primes_found)

    for k in range(1, len(primes)+1):
        if time.time() > deadline - 5:
            break
            
        P = set(primes[:k])
        
        # Precompute valid sums list
        valid_sums_list = []
        valid_sums_set = set()
        for s in range(2, MAX_SUM + 1):
            if get_prime_factors(s).issubset(P):
                valid_sums_list.append(s)
                valid_sums_set.add(s)
                
        if not valid_sums_list:
            continue

        found = False
        
        # Local Search with incremental update
        num_restarts = 300
        steps_per_restart = 1000
        
        for r in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            current = random.sample(range(1, MAX_VAL + 1), 12)
            
            # Initial bad count
            bad_count = 0
            for i in range(12):
                for j in range(i+1, 12):
                    if current[i] + current[j] not in valid_sums_set:
                        bad_count += 1
                        
            if bad_count == 0:
                found = True
                best_set = current
                best_k = k
                break
                
            no_improve = 0
            for step in range(steps_per_restart):
                if time.time() > deadline - 2:
                    break
                    
                i = random.randint(0, 11)
                old_val = current[i]
                
                # Constructive move
                j = random.randint(0, 11)
                while j == i:
                    j = random.randint(0, 11)
                    
                s = random.choice(valid_sums_list)
                new_val = s - current[j]
                
                if 1 <= new_val <= MAX_VAL and new_val != old_val:
                    # Check distinctness
                    is_distinct = True
                    for m in range(12):
                        if m != i and current[m] == new_val:
                            is_distinct = False
                            break
                            
                    if is_distinct:
                        # Calculate delta bad count
                        delta = 0
                        for m in range(12):
                            if m == i:
                                continue
                            # Remove old
                            if old_val + current[m] not in valid_sums_set:
                                delta -= 1
                            # Add new
                            if new_val + current[m] not in valid_sums_set:
                                delta += 1
                                
                        new_bad_count = bad_count + delta
                        
                        if new_bad_count < bad_count:
                            bad_count = new_bad_count
                            current[i] = new_val
                            no_improve = 0
                            
                            if bad_count == 0:
                                found = True
                                best_set = current
                                best_k = k
                                break
                        else:
                            no_improve += 1
                            
                if found:
                    break
                    
        if found:
            break
            
    if best_set is not None:
        return best_set
        
    # Fallback
    for _ in range(200):
        if time.time() > deadline - 1:
            break
        s = random.sample(range(1, MAX_VAL + 1), 12)
        sc = calc_raw_score(s)
        if sc < fallback_best_score:
            fallback_best_score = sc
            fallback_best_set = s
            
    return fallback_best_set
