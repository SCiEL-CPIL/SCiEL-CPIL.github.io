import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Phase 1: Generate powerful numbers up to a moderate limit for sum counting
    limit_small = 50000000  # 5 * 10^7
    
    # Generate powerful numbers using DFS with primes
    # Powerful: n = prod p_i^(e_i) where e_i >= 2
    # Equivalently: n = m^2 * k where k is squarefree and all primes in k appear in m
    # Actually: n is powerful iff for every prime p|n, p^2|n
    # So n = prod p_i^(2*f_i) where f_i >= 1, OR n=1
    
    # Better: n = a^2 * b where b is squarefree and rad(b) | a
    # Simplest: enumerate by prime factorization with exponents >= 2
    
    # Use the form: n = prod_{i} p_i^{e_i} where e_i >= 2
    # DFS over primes
    
    def gen_powerful(limit, max_prime_idx=None):
        """Generate all powerful numbers <= limit using DFS."""
        # Primes up to limit^(1/2) since smallest exponent is 2
        max_p = int(math.isqrt(limit)) + 1
        # Sieve for primes
        sieve = bytearray(b'\x01') * (max_p + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(math.isqrt(max_p)) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        primes = [i for i in range(2, max_p + 1) if sieve[i]]
        
        if max_prime_idx is not None:
            primes = primes[:max_prime_idx]
        
        results = [1]
        
        def dfs(idx, current):
            if idx >= len(primes):
                return
            p = primes[idx]
            # p^2, p^3, p^4, ... <= limit/current
            val = current * p * p
            while val <= limit:
                results.append(val)
                dfs(idx + 1, val)
                val *= p
        
        for i in range(len(primes)):
            p = primes[i]
            if p * p > limit:
                break
            # Start with p^2
            val = p * p
            while val <= limit:
                results.append(val)
                dfs(i + 1, val)
                val *= p
        
        return results
    
    # Generate powerful numbers up to limit_small
    pows_small = gen_powerful(limit_small)
    pows_small_set = set(pows_small)
    
    # Phase 2: Count sums using NumPy
    P = np.array(sorted(pows_small), dtype=np.int64)
    K = len(P)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # For large K, use blocked approach to avoid memory issues
    if K * K > 10**8:
        # Use blocking
        block_size = max(1, 10**8 // K)
        sum_counts = {}
        for i in range(0, K, block_size):
            end_i = min(i + block_size, K)
            block = P[i:end_i, None] + P[None, :]
            # Count unique sums in this block
            flat = block.ravel()
            unique, counts = np.unique(flat, return_counts=True)
            for s, c in zip(unique, counts):
                sum_counts[int(s)] = sum_counts.get(int(s), 0) + int(c)
    else:
        sums = P[:, None] + P[None, :]
        unique_sums, counts = np.unique(sums, return_counts=True)
        sum_counts = dict(zip(unique_sums.tolist(), counts.tolist()))
    
    # Get top candidate sums
    sorted_sums = sorted(sum_counts.items(), key=lambda x: -x[1])
    
    # Phase 3: For top candidates, count representations using a larger powerful number set
    # Generate powerful numbers up to max_candidate_N
    top_candidates = sorted_sums[:20]
    max_N_candidate = max(c[0] for c in top_candidates) if top_candidates else limit_small * 2
    
    # Generate powerful numbers up to max_N_candidate (larger limit)
    limit_large = min(max_N_candidate, 200000000)  # Cap at 2*10^8 for time
    pows_large = gen_powerful(limit_large)
    pows_large_set = set(pows_large)
    
    # For each candidate N, count actual representations
    best_N = None
    best_pairs = []
    best_count = 0
    
    for N_cand, approx_count in top_candidates:
        if time.time() > deadline - 20:
            break
        
        if N_cand > limit_large:
            continue
        
        # Count representations: a + b = N_cand, a <= b, a,b powerful
        pairs = []
        # Iterate through powerful numbers a <= N_cand/2
        for a in pows_large:
            if a > N_cand // 2:
                break
            b = N_cand - a
            if b < a:
                continue
            if b in pows_large_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N_cand
            best_pairs = pairs
    
    if best_N is None or best_count == 0:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': int(best_N), 'pairs': best_pairs}
