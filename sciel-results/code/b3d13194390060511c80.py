import time
import random
import math

def compute_metrics(A):
    """
    Computes max(|A+A|, |A*A|) for a list of distinct positive integers.
    Uses Python sets for speed with small n=20.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):  # j >= i to avoid duplicates, but we need all pairs
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
            if i != j:
                # Already covered by the symmetric pair when we swap i,j
                # Actually we need all ordered pairs for the set size
                pass
    # Actually, A+A means all sums a_i + a_j for all i,j (including i=j)
    # The set size is what matters, so we can just do all pairs
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Generate a rich pool of smooth/structured candidates
    def generate_pool():
        pool = set()
        # Small integers
        for i in range(1, 50):
            pool.add(i)
        # Powers of small primes
        for base in [2, 3, 5, 7, 11, 13, 17]:
            val = 1
            while val <= 10000:
                pool.add(val)
                val *= base
        # Products of two small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        for p1 in small_primes:
            for p2 in small_primes:
                prod = p1 * p2
                if prod <= 10000:
                    pool.add(prod)
                prod2 = prod * p1
                if prod2 <= 10000:
                    pool.add(prod2)
                prod3 = prod * p2
                if prod3 <= 10000:
                    pool.add(prod3)
        # Geometric-like sequences
        for base in [2, 3, 4, 5]:
            val = 1
            while val <= 10000:
                pool.add(val)
                val *= base
        # Filter
        pool = [x for x in pool if 1 <= x <= 10000]
        random.shuffle(pool)
        return pool
    
    pool = generate_pool()
    
    # Define diverse structural seeds
    seeds = [
        # Seed 1: Consecutive small integers
        list(range(1, 21)),
        # Seed 2: Powers of 2
        [2**i for i in range(20)],
        # Seed 3: Mixed powers
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36],
        # Seed 4: Geometric with ratio ~1.5
        [int(2 * (1.5**i)) for i in range(20)],
        # Seed 5: Smooth numbers clustered
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28],
        # Seed 6: Powers of 2 and 3 mixed
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96],
        # Seed 7: Small integers with some structure
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27],
        # Seed 8: Highly composite-like
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 24, 28, 30, 32, 36, 40],
    ]
    
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
    for restart_idx, (init_score, init_A) in enumerate(valid_seeds[:6]):  # Top 6 seeds
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
                candidates.add(random.choice(pool))
            
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
                    new1, new2 = random.sample(pool, 2)
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
