import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
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
    pool = sorted(list(pool))
    pool_size = len(pool)
    
    # Precompute smallest prime factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Map pool values to indices for fast lookup
    val_to_idx = {v: i for i, v in enumerate(pool)}
    
    # Precompute factors for all pairs in the pool
    pair_primes = [[None for _ in range(pool_size)] for _ in range(pool_size)]
    
    for i in range(pool_size):
        vi = pool[i]
        for j in range(i + 1, pool_size):
            vj = pool[j]
            s = vi + vj
            factors = set()
            n = s
            while n > 1:
                p = spf[n]
                factors.add(p)
                while n % p == 0:
                    n //= p
            f_tuple = tuple(factors)
            pair_primes[i][j] = f_tuple
            pair_primes[j][i] = f_tuple
            
    if time.time() > deadline - 5.0:
         return random.sample(pool, 12)
         
    # Key improvement: Try multiple strategies
    # Strategy 1: Use numbers that are multiples of a common factor M
    # If all a_i are multiples of M, then a_i + a_j is also a multiple of M
    # This constrains the prime factors to those of M plus those of (a_i+M)/M + (a_j+M)/M
    
    best_score = float('inf')
    best_set = None
    
    # Try different multipliers M
    # For each M, generate candidates that are multiples of M and 5-smooth
    for M in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48, 50, 60, 64, 72, 80, 90, 96, 100, 120, 128, 144, 160, 180, 192, 200, 240, 256, 288, 320, 360, 384, 400, 480, 512, 576, 640, 720, 768, 800, 960, 1024, 1152, 1280, 1440, 1536, 1600, 1920, 2048, 2304, 2560, 2880, 3072, 3200, 3840, 4096, 4608, 5120, 5760, 6144, 6400, 7680, 8192, 9216, 10240]:
        if M > 10000:
            break
        if time.time() > deadline - 3.0:
            break
            
        # Generate candidates: multiples of M that are 5-smooth
        candidates = [v for v in pool if v % M == 0]
        if len(candidates) < 12:
            continue
            
        # Filter to keep only the most promising candidates
        # Prefer smaller numbers to keep sums small
        candidates = sorted(candidates)[:min(len(candidates), 200)]
        cand_size = len(candidates)
        cand_to_pool_idx = {v: val_to_idx[v] for v in candidates}
        
        # Try to find 12 numbers from candidates
        # Use greedy + local search
        
        # Greedy initialization: pick numbers that minimize new prime factors
        current_set = []
        current_idx_set = []
        prime_counts = {}
        
        for _ in range(12):
            best_candidate = None
            best_score_candidate = float('inf')
            
            for v in candidates:
                if v in current_set:
                    continue
                
                # Compute new prime factors introduced
                new_primes = set()
                for other in current_set:
                    pidx = cand_to_pool_idx[v]
                    oidx = cand_to_pool_idx[other]
                    primes = pair_primes[pidx][oidx]
                    for p in primes:
                        if p not in prime_counts:
                            new_primes.add(p)
                
                total_primes = len(prime_counts) + len(new_primes)
                if total_primes < best_score_candidate:
                    best_score_candidate = total_primes
                    best_candidate = v
            
            if best_candidate is None:
                break
                
            current_set.append(best_candidate)
            current_idx_set.append(cand_to_pool_idx[best_candidate])
            
            # Update prime counts
            for i in range(len(current_set) - 1):
                pidx = current_idx_set[-1]
                oidx = current_idx_set[i]
                primes = pair_primes[pidx][oidx]
                for p in primes:
                    if p in prime_counts:
                        prime_counts[p] += 1
                    else:
                        prime_counts[p] = 1
        
        if len(current_set) < 12:
            continue
            
        current_score = len(prime_counts)
        
        # Local search: try to improve by swapping
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(12):
                old_val = current_set[i]
                old_idx = current_idx_set[i]
                
                for v in candidates:
                    if v == old_val or v in current_set:
                        continue
                    
                    new_idx = cand_to_pool_idx[v]
                    
                    # Compute delta
                    removed_primes = {}
                    added_primes = {}
                    
                    for k in range(12):
                        if k == i:
                            continue
                        other_idx = current_idx_set[k]
                        
                        # Removed primes
                        primes = pair_primes[old_idx][other_idx]
                        for p in primes:
                            removed_primes[p] = removed_primes.get(p, 0) + 1
                        
                        # Added primes
                        primes = pair_primes[new_idx][other_idx]
                        for p in primes:
                            added_primes[p] = added_primes.get(p, 0) + 1
                    
                    # Compute new score
                    new_score = len(prime_counts)
                    for p, cnt in removed_primes.items():
                        if p in prime_counts:
                            new_cnt = prime_counts[p] - cnt
                            if new_cnt <= 0:
                                new_score -= 1
                    for p, cnt in added_primes.items():
                        if p not in prime_counts or prime_counts[p] - removed_primes.get(p, 0) <= 0:
                            new_score += 1
                    
                    if new_score < current_score:
                        # Apply the swap
                        for k in range(12):
                            if k == i:
                                continue
                            other_idx = current_idx_set[k]
                            primes = pair_primes[old_idx][other_idx]
                            for p in primes:
                                prime_counts[p] -= 1
                                if prime_counts[p] == 0:
                                    del prime_counts[p]
                        
                        for k in range(12):
                            if k == i:
                                continue
                            other_idx = current_idx_set[k]
                            primes = pair_primes[new_idx][other_idx]
                            for p in primes:
                                if p in prime_counts:
                                    prime_counts[p] += 1
                                else:
                                    prime_counts[p] = 1
                        
                        current_set[i] = v
                        current_idx_set[i] = new_idx
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current_set[:]
                        break  # Restart loop since set changed
        
        if best_set is None or current_score < best_score:
            best_score = current_score
            best_set = current_set[:]
        
        # Early exit if we hit a very good score
        if best_score <= 3:
            break
    
    # Fallback: if we didn't find a good solution, use simulated annealing on the full pool
    if best_set is None or time.time() < deadline - 10.0:
        current_set = random.sample(pool, 12)
        current_idx_set = [val_to_idx[v] for v in current_set]
        
        prime_counts = {}
        for i in range(12):
            for j in range(i + 1, 12):
                idx_i = current_idx_set[i]
                idx_j = current_idx_set[j]
                primes = pair_primes[idx_i][idx_j]
                for p in primes:
                    if p in prime_counts:
                        prime_counts[p] += 1
                    else:
                        prime_counts[p] = 1
                        
        current_score = len(prime_counts)
        if best_set is None or current_score < best_score:
            best_score = current_score
            best_set = current_set[:]
        
        # Simulated annealing
        temp = 1.0
        cooling = 0.9999
        min_temp = 0.01
        
        while time.time() < deadline - 2.0:
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            old_idx = current_idx_set[idx]
            
            new_idx = random.randint(0, pool_size - 1)
            new_val = pool[new_idx]
            
            if new_idx == old_idx or new_val in current_set:
                continue
                
            for k in range(12):
                if k == idx:
                    continue
                other_idx = current_idx_set[k]
                primes = pair_primes[old_idx][other_idx]
                for p in primes:
                    prime_counts[p] -= 1
                    if prime_counts[p] == 0:
                        del prime_counts[p]
                        
            for k in range(12):
                if k == idx:
                    continue
                other_idx = current_idx_set[k]
                primes = pair_primes[new_idx][other_idx]
                for p in primes:
                    if p in prime_counts:
                        prime_counts[p] += 1
                    else:
                        prime_counts[p] = 1
                        
            current_set[idx] = new_val
            current_idx_set[idx] = new_idx
            new_score = len(prime_counts)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_set[:]
            else:
                delta = new_score - current_score
                accept_prob = math.exp(-delta / temp) if delta > 0 else 1.0
                
                if random.random() < accept_prob:
                    current_score = new_score
                else:
                    for k in range(12):
                        if k == idx:
                            continue
                        other_idx = current_idx_set[k]
                        primes = pair_primes[new_idx][other_idx]
                        for p in primes:
                            prime_counts[p] -= 1
                            if prime_counts[p] == 0:
                                del prime_counts[p]
                                
                    for k in range(12):
                        if k == idx:
                            continue
                        other_idx = current_idx_set[k]
                        primes = pair_primes[old_idx][other_idx]
                        for p in primes:
                            if p in prime_counts:
                                prime_counts[p] += 1
                            else:
                                prime_counts[p] = 1
                                
                    current_set[idx] = old_val
                    current_idx_set[idx] = old_idx
                    
            temp *= cooling
            if temp < min_temp:
                temp = min_temp
    
    return best_set if best_set is not None else random.sample(pool, 12)
