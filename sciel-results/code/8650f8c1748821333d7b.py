import time
import random
import math
import numpy as np

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    Uses NumPy for vectorization.
    """
    arr = np.array(A, dtype=np.int64)
    
    # Sumset A+A
    # Outer sum: arr[:, None] + arr[None, :]
    sums = arr[:, None] + arr[None, :]
    sum_set_size = len(np.unique(sums))
    
    # Product set A*A
    # Outer product: arr[:, None] * arr[None, :]
    prods = arr[:, None] * arr[None, :]
    prod_set_size = len(np.unique(prods))
    
    return max(sum_set_size, prod_set_size)

def is_smooth(n, max_prime_factor=50):
    """Check if n is smooth (all prime factors are <= max_prime_factor)."""
    if n <= 1:
        return True
    d = 2
    while d * d <= n and d <= max_prime_factor:
        while n % d == 0:
            n //= d
        d += 1
    if n > 1 and n > max_prime_factor:
        return False
    return True

def generate_smooth_candidates(count, max_val=10000, max_prime_factor=100):
    """Generate a list of 'smooth' numbers up to max_val."""
    candidates = set()
    # Add small integers
    for i in range(1, 100):
        candidates.add(i)
    
    # Add powers of 2, 3, 5, 7
    for base in [2, 3, 5, 7, 11, 13]:
        val = 1
        while val <= max_val:
            candidates.add(val)
            val *= base
    
    # Add products of small primes
    for p1 in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        for p2 in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            prod = p1 * p2
            if prod <= max_val:
                candidates.add(prod)
            # Add another factor
            for p3 in [2, 3, 5, 7, 11]:
                prod3 = prod * p3
                if prod3 <= max_val:
                    candidates.add(prod3)

    # Filter and limit
    smooth_list = [x for x in candidates if x <= max_val and is_smooth(x, max_prime_factor)]
    # Shuffle to ensure diversity in selection if we were to sample
    random.shuffle(smooth_list)
    return smooth_list

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 60s limit
    
    # Pre-generate a pool of "good" candidates
    # These are numbers likely to create product collisions (smooth numbers)
    # and sum collisions (small numbers)
    candidate_pool = generate_smooth_candidates(1000, 10000, 100)
    
    # Ensure 1 is in the pool
    if 1 not in candidate_pool:
        candidate_pool.append(1)
        
    # Seed 1: 1..20 (Minimizes sumset to 39, product set is moderate)
    seeds = [
        list(range(1, 21)),
        # Seed 2: Powers of 2 mixed with small odds (High product structure)
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 5, 7, 9, 11, 13],
        # Seed 3: Highly smooth set (Many collisions)
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
    ]
    
    best_A = None
    best_score = float('inf')
    
    for seed in seeds:
        # Ensure seed is valid (distinct, positive, <=10000)
        if len(set(seed)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in seed):
            continue
        
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    if best_A is None:
        # Fallback
        best_A = list(range(1, 21))
        best_score = compute_metrics(best_A)

    # Local Search with Simulated Annealing
    A = best_A[:]
    current_score = best_score
    
    # Parameters for Simulated Annealing
    T = 100.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    
    stagnant_count = 0
    max_stagnant = 500
    
    while time.time() < deadline and T > min_T:
        # Choose a random element to replace
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Generate candidates
        candidates = set()
        
        # 1. Random from smooth pool
        for _ in range(10):
            candidates.add(random.choice(candidate_pool))
            
        # 2. Neighbors of old_val
        for delta in range(-5, 6):
            nb = old_val + delta
            if 1 <= nb <= 10000:
                candidates.add(nb)
                
        # 3. Multiples/Divisors of old_val (if smooth)
        if old_val > 1:
            for d in range(2, 15):
                if old_val % d == 0:
                    candidates.add(old_val // d)
                candidates.add(old_val * d)
                
        # 4. Values that might create sum collisions with existing elements
        # If we replace A[idx] with x, we want x + A[j] to collide with existing sums.
        # This is hard to predict without full evaluation, so we rely on smoothness.
        
        A_set = set(A)
        A_set.discard(old_val)
        
        best_cand = None
        best_cand_score = current_score
        accepted_worse = False
        
        # Limit candidates to avoid spending too much time if the pool is huge
        # but here the pool is small and filtered.
        # Shuffle candidates to avoid bias
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for c in cand_list:
            if c in A_set:
                continue
            if c == old_val:
                continue
            if c < 1 or c > 10000:
                continue
            
            A[idx] = c
            s = compute_metrics(A)
            
            if s < best_cand_score:
                best_cand_score = s
                best_cand = c
                accepted_worse = False
            else:
                # Simulated Annealing: accept worse solution with probability
                delta = s - best_cand_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                
                if random.random() < prob:
                    best_cand_score = s
                    best_cand = c
                    accepted_worse = True

        if best_cand is not None:
            A[idx] = best_cand
            current_score = best_cand_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
                stagnant_count = 0
            else:
                stagnant_count += 1
                
            if not accepted_worse:
                T *= cooling_rate
        else:
            A[idx] = old_val
            stagnant_count += 1
            T *= cooling_rate
            
            # If stuck for a long time, try a more drastic move:
            # Replace 2 random elements with 2 random smooth numbers
            if stagnant_count > max_stagnant:
                stagnant_count = 0
                # Pick 2 indices
                i1, i2 = random.sample(range(20), 2)
                old1, old2 = A[i1], A[i2]
                
                new1 = random.choice(candidate_pool)
                new2 = random.choice(candidate_pool)
                
                # Ensure validity
                A_set_local = set(A)
                A_set_local.discard(old1)
                A_set_local.discard(old2)
                
                if new1 not in A_set_local and new2 not in A_set_local and new1 != new2:
                    A[i1] = new1
                    A[i2] = new2
                    s = compute_metrics(A)
                    if s < current_score:
                        current_score = s
                        if s < best_score:
                            best_score = s
                            best_A = A[:]
                    else:
                        # Revert
                        A[i1] = old1
                        A[i2] = old2
                        
    return best_A
