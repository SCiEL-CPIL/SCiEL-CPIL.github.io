import time
import random
import math

def compute_metrics(A):
    """
    Computes max(|A+A|, |A*A|) for a list of 20 distinct positive integers.
    """
    n = len(A)
    sums = set()
    prods = set()
    # Optimization: Since A+A and A*A are symmetric (a+b = b+a, a*b = b*a),
    # we only need to iterate i <= j.
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

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

def generate_smooth_candidates(max_val=10000, max_prime_factor=100):
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
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p1 in small_primes:
        for p2 in small_primes:
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
    random.shuffle(smooth_list)
    return smooth_list

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin: 50s out of 60s limit
    
    # Pre-generate a pool of "good" candidates
    # These are numbers likely to create product collisions (smooth numbers)
    # and sum collisions (small numbers)
    candidate_pool = generate_smooth_candidates(10000, 100)
    
    # Ensure 1 is in the pool
    if 1 not in candidate_pool:
        candidate_pool.append(1)
        
    # Define diverse structural seeds
    # Seed 1: Consecutive small integers
    seed_1 = list(range(1, 21))
    # Seed 2: Powers of 2 mixed with small odds (High product structure)
    seed_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 5, 7, 9, 11, 13]
    # Seed 3: Highly smooth set (Many collisions)
    seed_3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
    # Seed 4: Mixed powers
    seed_4 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
    # Seed 5: Geometric with ratio ~1.5
    seed_5 = [int(2 * (1.5**i)) for i in range(20)]
    # Seed 6: Powers of 2 and 3 mixed
    seed_6 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96]
    
    seeds = [seed_1, seed_2, seed_3, seed_4, seed_5, seed_6]
    
    # Validate and compute initial scores
    valid_seeds = []
    for seed in seeds:
        if len(set(seed)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in seed):
            continue
        score = compute_metrics(seed)
        valid_seeds.append((score, seed[:]))
    
    if not valid_seeds:
        return list(range(1, 21))
    
    valid_seeds.sort(key=lambda x: x[0])
    best_score, best_A = valid_seeds[0]
    
    # Multi-restart simulated annealing
    # We iterate over the top seeds to start from different basins
    for restart_idx, (init_score, init_A) in enumerate(valid_seeds[:6]):
        if time.time() > deadline:
            break
        
        A = init_A[:]
        current_score = init_score
        
        T = 50.0
        cooling_rate = 0.9995
        min_T = 0.5
        stagnant = 0
        max_stagnant = 300
        
        while time.time() < deadline and T > min_T:
            # Choose index to modify
            idx = random.randint(0, 19)
            old_val = A[idx]
            A_set = set(A)
            A_set.discard(old_val)
            
            # Generate diverse candidates
            candidates = set()
            
            # From pool
            for _ in range(8):
                candidates.add(random.choice(candidate_pool))
            
            # Neighbors
            for delta in range(-3, 4):
                nb = old_val + delta
                if 1 <= nb <= 10000:
                    candidates.add(nb)
            
            # Multiples and divisors
            if old_val > 1:
                for d in range(2, 8):
                    if old_val % d == 0:
                        candidates.add(old_val // d)
                    if old_val * d <= 10000:
                        candidates.add(old_val * d)
            
            # Filter valid candidates
            cand_list = [c for c in candidates if c not in A_set and 1 <= c <= 10000]
            random.shuffle(cand_list)
            
            best_cand = None
            best_cand_score = current_score
            
            for c in cand_list[:15]:  # Limit per iteration
                A[idx] = c
                s = compute_metrics(A)
                
                if s < best_cand_score:
                    best_cand_score = s
                    best_cand = c
                else:
                    # Accept worse with probability
                    delta_s = s - best_cand_score
                    if delta_s >= 0:
                        prob = math.exp(-delta_s / T)
                        if random.random() < prob:
                            best_cand_score = s
                            best_cand = c
            
            if best_cand is not None:
                A[idx] = best_cand
                current_score = best_cand_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
                    stagnant = 0
                else:
                    stagnant += 1
                T *= cooling_rate
            else:
                A[idx] = old_val
                stagnant += 1
                T *= cooling_rate
                
                if stagnant > max_stagnant:
                    stagnant = 0
                    # Drastic move: replace 2 elements
                    i1, i2 = random.sample(range(20), 2)
                    old1, old2 = A[i1], A[i2]
                    new1, new2 = random.sample(candidate_pool, 2)
                    A_set_local = set(A)
                    A_set_local.discard(old1)
                    A_set_local.discard(old2)
                    if new1 not in A_set_local and new2 not in A_set_local:
                        A[i1] = new1
                        A[i2] = new2
                        s = compute_metrics(A)
                        if s < current_score:
                            current_score = s
                            if s < best_score:
                                best_score = s
                                best_A = A[:]
                        else:
                            A[i1] = old1
                            A[i2] = old2
    
    return best_A
