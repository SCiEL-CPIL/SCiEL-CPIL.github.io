"""Combined program: structured powerful-number generation + frequency-based candidate identification."""
import time
import random
from math import isqrt
from collections import Counter
import numpy as np

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    r = isqrt(n)
    if r * r == n:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit as a^2 * b^3 where b is square-free."""
    powerful = set([1])
    max_b = int(limit ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_b
    sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    limit_sieve = isqrt(max_b) + 1
    for p in range(2, limit_sieve):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def _count_reps(N, powerful_list, powerful_set):
    """Count pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def _verify_powerful(n):
    """Verify n is powerful by checking factorization (for final validation)."""
    if n == 1:
        return True
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                cnt += 1
                temp //= d
            if cnt < 2:
                return False
        d += 1 if d == 2 else 2
    if temp > 1:
        return False  # remaining factor has exponent 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Generate powerful numbers up to a reasonable limit
    # We need a+b <= 10^18, so we can generate up to 10^18 but that's too many.
    # Generate up to 10^9 for a good balance of coverage and count.
    limit_pw = 10**9
    
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    elapsed = time.time() - start_time
    print(f"Generated {n_pw} powerful numbers up to {limit_pw} in {elapsed:.2f}s")
    
    if n_pw == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy 1: Frequency-based candidate identification via random pairwise sum sampling
    random.seed(42)
    sum_count = Counter()
    
    # Sample many random pairs to identify high-frequency sums
    sample_size = 20_000_000
    arr = np.array(powerful, dtype=np.int64)
    
    for _ in range(sample_size):
        if time.time() > deadline - 40:
            break
        # Vectorized sampling for speed
        idx1 = np.random.randint(0, n_pw, size=1000)
        idx2 = np.random.randint(0, n_pw, size=1000)
        sums = arr[idx1] + arr[idx2]
        for s in sums:
            if s <= 10**18 and s >= 2:
                sum_count[int(s)] += 1
    
    elapsed = time.time() - start_time
    print(f"Sampling done in {elapsed:.2f}s, found {len(sum_count)} unique sums")
    
    # Get top candidates by frequency
    top_candidates = sum_count.most_common(100)
    
    # Verify each top candidate with full scan
    for cand_n, freq in top_candidates:
        if time.time() > deadline - 15:
            break
        if cand_n > 10**18 or cand_n < 2:
            continue
        count, pairs = _count_reps(cand_n, powerful, powerful_set)
        if count > best_count:
            best_count = count
            best_N = cand_n
            best_pairs = pairs
            print(f"New best: N={cand_n}, count={count}")
    
    # Strategy 2: Also try structured approach - N = R * M where R is powerful
    # and M has many sum-of-two-squares representations
    if time.time() < deadline - 20:
        R_candidates = [1, 8, 27, 125, 343, 1331, 2197, 4913, 15625, 28561, 42875, 10648, 27000, 3375, 421875, 9261000]
        primes_4k1 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317]
        
        for R in R_candidates:
            if time.time() > deadline - 10:
                break
            
            M_max = 10**18 // R
            if M_max < 2:
                continue
            
            # Try M = (prod of first k primes_4k1)^2
            product = 1
            for k, p in enumerate(primes_4k1):
                new_product = product * p
                M_cand = new_product * new_product
                if M_cand > M_max:
                    break
                
                N_cand = R * M_cand
                if N_cand < 2 or N_cand > 10**18:
                    break
                
                # Count representations
                count, pairs = _count_reps(N_cand, powerful, powerful_set)
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = pairs
                    print(f"Structured new best: N={N_cand}, count={count}")
                
                product = new_product
            
            # Also try M = 4 * (prod of first k primes_4k1)^2
            product = 1
            for k, p in enumerate(primes_4k1):
                new_product = product * p
                M_cand = 4 * new_product * new_product
                if M_cand > M_max:
                    break
                
                N_cand = R * M_cand
                if N_cand < 2 or N_cand > 10**18:
                    break
                
                count, pairs = _count_reps(N_cand, powerful, powerful_set)
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = pairs
                    print(f"Structured(4x) new best: N={N_cand}, count={count}")
                
                product = new_product
    
    # Strategy 3: Additional random search if time permits
    if time.time() < deadline - 5 and best_count < 30:
        random.seed(123)
        for _ in range(50000):
            if time.time() > deadline - 3:
                break
            idx1 = random.randint(0, n_pw - 1)
            idx2 = random.randint(0, n_pw - 1)
            n_candidate = powerful[idx1] + powerful[idx2]
            
            if n_candidate > 10**18 or n_candidate < 2:
                continue
            
            count, pairs = _count_reps(n_candidate, powerful, powerful_set)
            if count > best_count:
                best_count = count
                best_N = n_candidate
                best_pairs = pairs
    
    # Final verification of best result
    if best_N is not None and best_count > 0:
        verified_pairs = []
        for a, b in best_pairs:
            if a + b == best_N and a <= b and a >= 1 and b >= 1:
                # Quick check: both should be in powerful_set or verify
                if a in powerful_set and b in powerful_set:
                    verified_pairs.append([a, b])
                else:
                    # Verify by factorization for numbers outside our generated set
                    if _verify_powerful(a) and _verify_powerful(b):
                        verified_pairs.append([a, b])
        
        if len(verified_pairs) > 0:
            return {'N': best_N, 'pairs': verified_pairs}
    
    # Fallback
    return {'N': 2, 'pairs': [[1, 1]]}
