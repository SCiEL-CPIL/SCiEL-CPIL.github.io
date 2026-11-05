import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 90s kill
    
    # Precompute SPF up to 20000
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
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    # Precompute prime factors for all numbers up to 20000
    pf = [set() for _ in range(MAX_SUM + 1)]
    for n in range(2, MAX_SUM + 1):
        pf[n] = get_prime_factors(n)
    
    # Generate pool of 5-smooth numbers (primes 2, 3, 5) up to 10000
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
    
    # Precompute sum factors for all pairs in pool
    sum_factors = {}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            s = pool[i] + pool[j]
            if s not in sum_factors:
                sum_factors[s] = pf[s]
    
    def evaluate_set(numbers):
        prime_union = set()
        for i in range(12):
            for j in range(i + 1, 12):
                s = numbers[i] + numbers[j]
                prime_union.update(sum_factors.get(s, pf[s]))
        return len(prime_union)
    
    # Lattice-based construction: a_i = u_i * P + v_i * Q
    # Strategy: choose P and Q such that P+Q, P-Q, etc. have few prime factors
    # Try small P, Q values and find 12 distinct combinations
    
    def lattice_search():
        best_lattice = None
        best_lattice_score = float('inf')
        
        # Try various (P, Q) pairs
        # P and Q should be small to keep values <= 10000
        # We want a_i = u_i * P + v_i * Q where u_i, v_i are small non-negative integers
        # and all a_i are distinct and <= 10000
        
        P_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63, 70, 72, 75, 80, 84, 90, 96, 100, 105, 108, 112, 120]
        Q_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63, 70, 72, 75, 80, 84, 90, 96, 100, 105, 108, 112, 120]
        
        for P in P_candidates:
            if time.time() > deadline - 10:
                break
            for Q in Q_candidates:
                if P == Q:
                    continue
                # Generate all values u*P + v*Q <= 10000
                values = set()
                for u in range(0, 10000 // P + 1):
                    for v in range(0, (10000 - u * P) // Q + 1):
                        val = u * P + v * Q
                        if 1 <= val <= 10000:
                            values.add(val)
                
                if len(values) < 12:
                    continue
                
                values_list = list(values)
                random.shuffle(values_list)
                
                # Try to find 12 values from this lattice that minimize prime count
                # Use a greedy approach
                best_subset = None
                best_subset_score = float('inf')
                
                for trial in range(20):
                    if time.time() > deadline - 10:
                        break
                    
                    current = []
                    used = set()
                    
                    # Pick first value randomly
                    first = random.choice(values_list[:min(100, len(values_list))])
                    current.append(first)
                    used.add(first)
                    
                    # Greedily add remaining 11
                    for _ in range(11):
                        best_cand = None
                        best_score = float('inf')
                        
                        # Sample candidates
                        avail = [x for x in values_list if x not in used]
                        sample_size = min(100, len(avail))
                        candidates = random.sample(avail, sample_size)
                        
                        for cand in candidates:
                            # Count new primes
                            new_primes = set()
                            for existing in current:
                                s = cand + existing
                                new_primes.update(pf[s])
                            
                            existing_primes = set()
                            for i in range(len(current)):
                                for j in range(i + 1, len(current)):
                                    existing_primes.update(pf[current[i] + current[j]])
                            
                            total = len(existing_primes | new_primes)
                            if total < best_score:
                                best_score = total
                                best_cand = cand
                        
                        if best_cand is not None:
                            current.append(best_cand)
                            used.add(best_cand)
                    
                    if len(current) == 12:
                        score = evaluate_set(current)
                        if score < best_subset_score:
                            best_subset_score = score
                            best_subset = current[:]
                
                if best_subset is not None and best_subset_score < best_lattice_score:
                    best_lattice_score = best_subset_score
                    best_lattice = best_subset[:]
                
                if best_lattice_score <= 3:
                    break
            if best_lattice_score <= 3:
                break
        
        return best_lattice, best_lattice_score
    
    # Run lattice search
    lattice_set, lattice_score = lattice_search()
    
    # Also run the smooth-number-based greedy
    def smooth_greedy():
        best_greedy = None
        best_greedy_score = float('inf')
        
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            current = []
            used = set()
            
            # Pick first element from lower half of pool
            first = random.choice(pool[:len(pool)//2])
            current.append(first)
            used.add(first)
            
            for _ in range(11):
                best_cand = None
                best_score = float('inf')
                
                avail = [x for x in pool if x not in used]
                sample_size = min(150, len(avail))
                candidates = random.sample(avail, sample_size)
                
                for cand in candidates:
                    new_primes = set()
                    for existing in current:
                        s = cand + existing
                        new_primes.update(sum_factors.get(s, pf[s]))
                    
                    existing_primes = set()
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            existing_primes.update(sum_factors.get(current[i] + current[j], pf[current[i] + current[j]]))
                    
                    total = len(existing_primes | new_primes)
                    if total < best_score:
                        best_score = total
                        best_cand = cand
                
                if best_cand is not None:
                    current.append(best_cand)
                    used.add(best_cand)
            
            if len(current) == 12:
                score = evaluate_set(current)
                if score < best_greedy_score:
                    best_greedy_score = score
                    best_greedy = current[:]
        
        return best_greedy, best_greedy_score
    
    smooth_set, smooth_score = smooth_greedy()
    
    # Pick the better starting point
    if lattice_set is not None and (smooth_set is None or lattice_score <= smooth_score):
        current_set = lattice_set[:]
        current_score = lattice_score
    else:
        current_set = smooth_set[:] if smooth_set is not None else random.sample(pool, 12)
        current_score = smooth_score if smooth_set is not None else evaluate_set(current_set)
    
    best_score = current_score
    best_set = current_set[:]
    
    # Local search with simulated annealing
    temp = 2.0
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 3000
    
    for step in range(300000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Single element swap
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
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
            
            revert_vals = (idx1, idx2, old_val1, old_val2)
        else:
            # Single element replacement with nearby value
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
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
                    current_set[idx] = old_val
                else:
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
                new_val = random.choice(pool)
                if new_val not in current_set:
                    current_set[idx] = new_val
            current_score = evaluate_set(current_set)
    
    if best_set is None:
        return list(range(1, 13))
        
    return best_set
