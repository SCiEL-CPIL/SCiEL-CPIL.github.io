# EVOLVE-BLOCK-START
import time
import math
import random

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit as a sorted list."""
    limit_cube_root = int(round(limit ** (1/3))) + 2
    while limit_cube_root ** 3 > limit:
        limit_cube_root -= 1
    while (limit_cube_root + 1) ** 3 <= limit:
        limit_cube_root += 1
    
    max_prime = int(math.isqrt(limit_cube_root)) + 1
    if max_prime < 2:
        max_prime = 2
    is_prime = [True] * (max_prime + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_prime)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_prime+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_prime+1) if is_prime[i]]
    
    sq_free = [True] * (limit_cube_root + 1)
    sq_free[0] = False
    for p in primes:
        p2 = p * p
        if p2 > limit_cube_root:
            break
        for j in range(p2, limit_cube_root + 1, p2):
            sq_free[j] = False
    
    square_free_list = [i for i in range(1, limit_cube_root + 1) if sq_free[i]]
    
    powerful_set = set()
    for y in square_free_list:
        y3 = y * y * y
        if y3 > limit:
            continue
        max_x = int(math.isqrt(limit // y3))
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerful_set.add(n)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s
    
    # Generate powerful numbers
    limit = 30_000_000  # 30 million
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    n_pow = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        # Fallback: just return a small valid result
        pairs = [[1, 1]]
        return {'N': 2, 'pairs': pairs}
    
    def count_reps(N):
        """Count pairs (a,b) with a<=b, a+b=N, a,b in powerful."""
        count = 0
        for a in powerful:
            if a > N // 2:
                break
            if (N - a) in powerful_set:
                count += 1
        return count
    
    best_N = 2
    best_count = 0
    
    # Generate candidate N values
    random.seed(12345)
    candidate_Ns = set()
    
    # Sample pairs of powerful numbers to get candidate N values
    sample_size = min(n_pow, 8000)
    if sample_size >= 2:
        indices = random.sample(range(n_pow), sample_size)
        for i in indices:
            a = powerful[i]
            # Sample b >= a
            if i < n_pow - 1:
                num_b = min(30, n_pow - i)
                b_indices = random.sample(range(i, n_pow), num_b)
            else:
                b_indices = [i]
            for j in b_indices:
                b = powerful[j]
                N = a + b
                if 2 <= N <= 10**18:
                    candidate_Ns.add(N)
    
    # Add some structured candidates
    for N in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        if N <= limit:
            candidate_Ns.add(N)
    
    # Sort candidates to evaluate in some order
    candidates = sorted(candidate_Ns)
    
    # Evaluate candidates
    for N in candidates:
        if time.time() - start_time > DEADLINE - 25:
            break
        if N < 2 or N > 10**18:
            continue
        count = count_reps(N)
        if count > best_count:
            best_count = count
            best_N = N
    
    # Local search around best_N
    if time.time() - start_time < DEADLINE - 15:
        for delta in range(1, 500):
            if time.time() - start_time > DEADLINE - 5:
                break
            for N in [best_N + delta, best_N - delta]:
                if N < 2 or N > 10**18:
                    continue
                count = count_reps(N)
                if count > best_count:
                    best_count = count
                    best_N = N
    
    # Extract pairs for best_N
    pairs = [[a, best_N - a] for a in powerful if a <= best_N - a and (best_N - a) in powerful_set]
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
