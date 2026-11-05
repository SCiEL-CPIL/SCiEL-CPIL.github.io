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
    
    # Precompute prime factors for all sums in pool
    # pool_size ~ 300, so ~45k pairs, sums up to 20000
    # Precompute factor sets for all possible sums
    sum_factors = {}
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            s = pool[i] + pool[j]
            if s not in sum_factors:
                n = s
                f = set()
                while n > 1:
                    p = spf[n]
                    f.add(p)
                    while n % p == 0:
                        n //= p
                sum_factors[s] = f
    
    def evaluate_set(numbers):
        prime_union = set()
        for i in range(12):
            for j in range(i + 1, 12):
                s = numbers[i] + numbers[j]
                prime_union.update(sum_factors.get(s, set()))
        return len(prime_union)
    
    # Smart initial construction: greedy with prime constraint
    def greedy_construct():
        # Try to build a set where all pairwise sums have few prime factors
        # Start with small numbers that have smooth sums
        candidates = sorted(pool, key=lambda x: len(get_prime_factors(x)))
        
        # Try multiple random starting points
        best_greedy = None
        best_greedy_score = float('inf')
        
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Randomly select first few elements
            current = []
            used = set()
            
            # Pick first element randomly from lower half
            first = random.choice(candidates[:len(candidates)//2])
            current.append(first)
            used.add(first)
            
            # Greedily add elements that minimize new primes
            for _ in range(11):
                best_candidate = None
                best_new_primes = float('inf')
                
                # Sample candidates to speed up
                sample_size = min(200, len(pool) - len(used))
                candidates_sample = random.sample([x for x in pool if x not in used], sample_size)
                
                for cand in candidates_sample:
                    # Count new primes introduced
                    new_primes = set()
                    for existing in current:
                        s = cand + existing
                        f = sum_factors.get(s, set())
                        new_primes.update(f)
                    
                    # Also count primes from existing pairs
                    existing_primes = set()
                    for i in range(len(current)):
                        for j in range(i+1, len(current)):
                            s = current[i] + current[j]
                            existing_primes.update(sum_factors.get(s, set()))
                    
                    total = len(existing_primes | new_primes)
                    
                    if total < best_new_primes:
                        best_new_primes = total
                        best_candidate = cand
                
                if best_candidate is not None:
                    current.append(best_candidate)
                    used.add(best_candidate)
            
            score = evaluate_set(current)
            if score < best_greedy_score:
                best_greedy_score = score
                best_greedy = current[:]
        
        return best_greedy, best_greedy_score
    
    # Run greedy construction
    current_set, current_score = greedy_construct()
    
    if current_set is None:
        current_set = random.sample(pool, 12)
        current_score = evaluate_set(current_set)
    
    best_score = current_score
    best_set = current_set[:]
    
    # Local search with multiple move types
    temp = 2.0
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 5000
    
    for step in range(500000):
        if time.time() > deadline:
            break
        
        # Choose move type: 70% single swap, 20% double swap, 10% insert/delete
        move_type = random.random()
        
        if move_type < 0.7:
            # Single element swap
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Pick new value from pool, prefer values with smooth sums
            new_val = random.choice(pool)
            if new_val == old_val or new_val in current_set:
                continue
            
            current_set[idx] = new_val
            new_score = evaluate_set(current_set)
            
        elif move_type < 0.9:
            # Double swap
            idx1, idx2 = random.sample(range(12), 2)
            old_val1, old_val2 = current_set[idx1], current_set[idx2]
            
            new_val1, new_val2 = random.sample(pool, 2)
            if new_val1 in current_set or new_val2 in current_set:
                continue
            if new_val1 == old_val1 and new_val2 == old_val2:
                continue
            
            current_set[idx1] = new_val1
            current_set[idx2] = new_val2
            new_score = evaluate_set(current_set)
            
            # Store for potential revert
            revert_vals = (idx1, idx2, old_val1, old_val2)
        else:
            # Single element replacement with nearby value
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Pick a value that's close in magnitude
            candidates_near = [x for x in pool if x not in current_set and abs(x - old_val) < 1000]
            if not candidates_near:
                candidates_near = [x for x in pool if x not in current_set]
            
            new_val = random.choice(candidates_near)
            if new_val == old_val:
                continue
            
            current_set[idx] = new_val
            new_score = evaluate_set(current_set)
        
        # Accept/reject
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
                if best_score <= 3:
                    break
        else:
            delta = new_score - current_score
            accept_prob = math.exp(-delta / max(temp, 0.01))
            
            if random.random() < accept_prob:
                current_score = new_score
                no_improve += 1
            else:
                # Revert
                if move_type < 0.7 or move_type >= 0.9:
                    # Single swap
                    current_set[idx] = old_val
                else:
                    # Double swap
                    i1, i2, v1, v2 = revert_vals
                    current_set[i1] = v1
                    current_set[i2] = v2
                no_improve += 1
        
        temp *= cooling
        if temp < 0.05:
            temp = 0.05
        
        if no_improve > max_no_improve:
            # Restart with perturbation
            no_improve = 0
            temp = 2.0
            for _ in range(5):
                idx = random.randint(0, 11)
                old_val = current_set[idx]
                new_val = random.choice(pool)
                if new_val not in current_set:
                    current_set[idx] = new_val
            current_score = evaluate_set(current_set)
    
    if best_set is None:
        return list(range(1, 13))
        
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# EVOLVE-BLOCK-END
