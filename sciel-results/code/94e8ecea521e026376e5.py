import time
import random
import math
import numpy as np

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    A = np.array(A, dtype=np.int64)
    n = len(A)
    sums = A[:, None] + A[None, :]
    prods = A[:, None] * A[None, :]
    sum_set = len(set(sums.ravel().tolist()))
    prod_set = len(set(prods.ravel().tolist()))
    return max(sum_set, prod_set)

def is_smooth(n, primes, max_exp):
    """Check if n is a product of primes with exponents <= max_exp."""
    for p in primes:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e > max_exp:
            return False
    return n == 1

def generate_smooth_candidates(max_val=10000, primes=(2, 3, 5, 7), max_exp=12):
    """Generate all smooth numbers up to max_val."""
    pool = set()
    def gen(idx, current):
        if idx == len(primes):
            if 1 <= current <= max_val:
                pool.add(current)
            return
        p = primes[idx]
        val = current
        for _ in range(max_exp + 1):
            if val > max_val:
                break
            gen(idx + 1, val)
            val *= p
    gen(0, 1)
    return sorted(pool)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    # Generate structured pool with multiplicative richness
    pool = generate_smooth_candidates(10000, (2, 3, 5, 7), max_exp=12)
    pool_set = set(pool)
    
    # Also include some small integers for sumset balance
    small_ints = list(range(1, 51))
    combined_pool = sorted(set(pool) | set(small_ints))
    combined_set = set(combined_pool)
    
    # Multiple diverse initial seeds
    seeds = []
    
    # Seed 1: Small integers (good sumset)
    seeds.append(list(range(1, 21)))
    
    # Seed 2: Powers of 2 and 3 mixed (good product set)
    p23 = [2**a * 3**b for a in range(0, 14) for b in range(0, 14) if 2**a * 3**b <= 10000]
    p23 = sorted(set(p23))
    if len(p23) >= 20:
        step = max(1, len(p23) // 20)
        seed2 = [p23[i * step] for i in range(20)]
        if len(set(seed2)) == 20:
            seeds.append(seed2)
    
    # Seed 3: Smooth numbers from pool, spread out
    if len(pool) >= 20:
        step = max(1, len(pool) // 20)
        seed3 = [pool[i * step] for i in range(20)]
        if len(set(seed3)) == 20:
            seeds.append(seed3)
    
    # Seed 4: Mix of small and structured
    seed4 = list(range(1, 11))
    remaining_pool = [x for x in pool if x > 10]
    if len(remaining_pool) >= 10:
        step = max(1, len(remaining_pool) // 10)
        seed4 += [remaining_pool[i * step] for i in range(10)]
        seed4 = sorted(set(seed4))
        if len(seed4) == 20:
            seeds.append(seed4)
    
    # Seed 5: Geometric progression in log space
    try:
        log_min = math.log(2)
        log_max = math.log(10000)
        log_step = (log_max - log_min) / 19
        seed5 = []
        for i in range(20):
            val = int(round(math.exp(log_min + i * log_step)))
            val = max(1, min(10000, val))
            seed5.append(val)
        seed5 = sorted(set(seed5))
        if len(seed5) == 20:
            seeds.append(seed5)
    except:
        pass
    
    # Seed 6: Pure 2-smooth (powers of 2)
    p2 = [2**a for a in range(1, 15) if 2**a <= 10000]
    if len(p2) >= 20:
        seeds.append(p2[:20])
    
    # Seed 7: Mixed primes - 2^a * 3^b with specific patterns
    seed7 = []
    for a in range(0, 10):
        for b in range(0, 10):
            v = 2**a * 3**b
            if 1 <= v <= 10000:
                seed7.append(v)
    seed7 = sorted(set(seed7))
    if len(seed7) >= 20:
        # Pick a diverse subset
        step = max(1, len(seed7) // 20)
        seed7_sub = [seed7[i * step] for i in range(20)]
        if len(set(seed7_sub)) == 20:
            seeds.append(seed7_sub)
    
    # Evaluate all seeds
    best_A = None
    best_score = float('inf')
    
    for seed in seeds:
        if len(set(seed)) != 20 or any(x < 1 or x > 10000 for x in seed):
            continue
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = compute_metrics(best_A)
    
    # Incremental metric computation for speed
    def compute_metrics_fast(A_list):
        """Faster metric computation using sets directly."""
        sums = set()
        prods = set()
        n = len(A_list)
        for i in range(n):
            for j in range(n):
                sums.add(A_list[i] + A_list[j])
                prods.add(A_list[i] * A_list[j])
        return max(len(sums), len(prods))
    
    # Local search with incremental updates
    A = best_A[:]
    current_score = best_score
    A_set = set(A)
    
    # Pre-compute current sums and prods
    def get_sums_prods(arr):
        sums = set()
        prods = set()
        n = len(arr)
        for i in range(n):
            for j in range(n):
                sums.add(arr[i] + arr[j])
                prods.add(arr[i] * arr[j])
        return sums, prods
    
    current_sums, current_prods = get_sums_prods(A)
    current_sum_count = len(current_sums)
    current_prod_count = len(current_prods)
    
    # Swap-based local search with structured candidates
    T = 30.0
    T_min = 0.01
    alpha = 0.9997
    
    no_improve_count = 0
    max_no_improve = 3000
    
    # Candidate pool for replacements - focus on smooth numbers
    candidate_list = combined_pool
    
    while time.time() < deadline and T > T_min:
        # Choose one index to replace (single-element move for speed)
        i = random.randint(0, 19)
        old_vi = A[i]
        
        # Generate new candidate biased towards structured pool
        r = random.random()
        if r < 0.6:
            new_vi = random.choice(candidate_list)
        elif r < 0.8:
            # Neighbor in value space
            delta = random.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10])
            new_vi = max(1, min(10000, old_vi + delta))
        else:
            new_vi = random.randint(1, 10000)
        
        # Check distinctness
        if new_vi == old_vi:
            continue
        if new_vi in A_set:
            continue
        
        # Apply replacement
        A[i] = new_vi
        A_set.discard(old_vi)
        A_set.add(new_vi)
        
        # Compute new metric
        new_score = compute_metrics_fast(A)
        
        # Acceptance
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            A[i] = old_vi
            A_set.discard(new_vi)
            A_set.add(old_vi)
            no_improve_count += 1
        
        T *= alpha
        
        # Restart if stuck too long
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Perturb: replace 2-3 random elements
            for _ in range(random.randint(2, 3)):
                idx = random.randint(0, 19)
                old_val = A[idx]
                new_val = random.choice(candidate_list)
                if new_val != old_val and new_val not in A_set:
                    A[idx] = new_val
                    A_set.discard(old_val)
                    A_set.add(new_val)
            current_score = compute_metrics_fast(A)
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        
        # Time check
        if time.time() > deadline - 3.0:
            break
    
    return best_A
