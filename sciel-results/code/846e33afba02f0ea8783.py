import time
import numpy as np
from math import isqrt

def _is_square_free(n, primes):
    """Check if n is square-free using precomputed primes."""
    if n == 1:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
    return True

def _sieve_primes(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0] = 0
    is_p[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number n satisfies: for every prime p|n, p^2|n.
    Equivalently, n = u^2 * v^3 where v is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # v must be square-free and v^3 <= limit
    max_v = int(round(limit ** (1.0 / 3.0))) + 2
    if max_v < 1:
        max_v = 1

    # Primes up to sqrt(max_v) for square-free check
    sieve_limit = isqrt(max_v) + 2
    if sieve_limit < 2:
        sieve_limit = 2
    primes = _sieve_primes(sieve_limit)
    
    for v in range(1, max_v + 1):
        if not _is_square_free(v, primes):
            continue
            
        v3 = v * v * v
        if v3 > limit:
            continue
            
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def _is_powerful(n):
    """
    Check if n is powerful using exact factorization.
    """
    if n == 1:
        return True
    if n <= 0:
        return False
    
    t = n
    if t % 2 == 0:
        exp = 0
        while t % 2 == 0:
            t //= 2
            exp += 1
        if exp == 1:
            return False
            
    p = 3
    while p * p <= t:
        if t % p == 0:
            exp = 0
            while t % p == 0:
                t //= p
                exp += 1
            if exp == 1:
                return False
        p += 2
        
    if t > 1:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will try to maximize M (limit for powerful numbers)
    # such that K^2 operations are feasible.
    # K ~ 1.75 * M^(1/3)
    # K^2 ~ 3 * M^(2/3)
    # For M=2e8, K~7400, K^2~5.5e7. 
    # NumPy can handle this.
    
    limits_to_try = [
        200_000_000,
        100_000_000,
        50_000_000
    ]
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    for limit in limits_to_try:
        if time.time() > deadline - 20:
            break
            
        try:
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 15:
                continue
                
            K = len(powerful)
            if K < 2:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Generate all sums P[i] + P[j] for i <= j
            # To save memory, we can do this in chunks if K is large, 
            # but for K~7000, total pairs ~25M, 200MB. 
            # Let's try to do it in one shot or two chunks.
            
            sums = []
            chunk_size = 5000
            for i in range(0, K, chunk_size):
                end_i = min(i + chunk_size, K)
                # For each i in [i, end_i), j ranges from i to K
                # This is a bit tricky to vectorize perfectly for i<=j without duplicates or complex masks.
                # Simpler: Just generate all i, j and then filter i<=j? 
                # Or generate upper triangle.
                
                # Let's just generate all sums for a block of i's against all j's >= i?
                # Actually, numpy broadcasting:
                # rows = pow_arr[i:end_i]
                # cols = pow_arr[i:] # This changes size for each i, inefficient.
                
                # Better: 
                # We want sums of pow_arr[a] + pow_arr[b] where a <= b.
                # We can iterate a from 0 to K-1.
                # sums_a = pow_arr[a] + pow_arr[a:]
                # This is O(K^2) total operations.
                # For K=7000, 25M additions.
                # Doing this in a Python loop is slow (7000 iterations).
                # But each iteration is a NumPy vector op.
                # 7000 * (vector add + append) might be slow due to overhead.
                
                # Alternative: 
                # Create a full KxK matrix? No, 7000x7000x8 = 392MB.
                # Then take upper triangle?
                # This is memory intensive but fast.
                
                # Let's stick to the loop but optimize it.
                # Actually, we can use a dictionary for counting if we process in chunks of i.
                
                pass # We will use a different counting strategy below.
                
            # Strategy: Use a dictionary to count sums.
            # Iterate i from 0 to K-1.
            # For each i, compute sums = pow_arr[i] + pow_arr[i:]
            # Update counts.
            # This is O(K^2) time, O(K) space per step (plus dict).
            # 7000 steps, each ~3500 elements avg.
            # Total 25M updates.
            # Python dict updates are fast, but 25M might take 5-10s.
            
            counts = {}
            for i in range(K):
                if time.time() > deadline - 10:
                    break
                a = pow_arr[i]
                # Vectorized sum
                b_part = pow_arr[i:]
                sums = a + b_part
                # Count unique sums in this chunk
                unique_sums, counts_s = np.unique(sums, return_counts=True)
                for s, c in zip(unique_sums, counts_s):
                    counts[int(s)] = counts.get(int(s), 0) + int(c)
            
            if not counts:
                continue
                
            # Find N with max count
            best_N, best_count = max(counts.items(), key=lambda x: x[1])
            
            if best_N > 10**18:
                continue
                
            # Extract pairs
            pow_set = set(powerful)
            pairs = []
            half_N = best_N // 2
            # a must be <= b, so a <= N/2
            # Find index of last powerful number <= N/2
            end_idx = np.searchsorted(pow_arr, half_N, side='right')
            
            for a in pow_arr[:end_idx]:
                b = best_N - a
                if b in pow_set:
                    # Verify powerful (should be true)
                    if _is_powerful(int(a)) and _is_powerful(int(b)):
                        pairs.append([int(a), int(b)])
                        
            if pairs and len(pairs) > len(best_result['pairs']):
                best_result = {'N': int(best_N), 'pairs': pairs}
                
        except Exception as e:
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
